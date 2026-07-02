# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# Convert the full 28-layer Qwen3-TTS Talker decoder stack to ncnn via pnnx.
#
# Strategy (validated by convert_talker_poc.py + verify_mrope_collapse.py):
#   - The Talker's interleaved mRoPE provably collapses to a single cos/sin table
#     in the TTS text path (all 3 position rows equal). We monkeypatch
#     apply_multimodal_rotary_pos_emb -> plain rotate_half RoPE (bit-identical),
#     so the exported graph is a clean RMSNorm/Gemm/RotaryEmbed/SDPA/MLP stack
#     with none of the Crop/CopyTo tangle the raw mRoPE produced.
#   - We wrap Qwen3TTSTalkerModel's layer loop with tensor-only I/O matching the
#     shipped decoder contract: in0=inputs_embeds (H x S), in1=additive causal
#     mask (1,1,S,S), in2=cos (1,S,head_dim), in3=sin (1,S,head_dim). No KV cache
#     during trace; kv_cache is patched into the .param afterwards (pnnx does not
#     emit it) via add_kvcache.py.
#   - Output: hidden states after final RMSNorm (out0). The codec_head (1024->3072)
#     is exported separately as a tiny proj_out net (see convert_talker_head.py).
#
# Usage:
#   python convert_talker.py --model models/Qwen3-TTS-12Hz-0.6B-Base \
#       --outdir tools/qwen3_tts/pnnx_out --seq 8

import argparse
import os
import torch
import torch.nn as nn


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def plain_rope(q, k, cos, sin, mrope_section, mrope_interleaved=False, unsqueeze_dim=1):
    """Drop-in replacement for apply_multimodal_rotary_pos_emb.

    Accepts the reduced single-table cos/sin of shape (B, S, head_dim) (or the
    original (3,B,S,head_dim) — we take row 0, which the collapse proof shows is
    identical). Applies standard rotate_half RoPE. Proven bit-exact vs the real
    interleaved mRoPE by verify_mrope_collapse.py.
    """
    if cos.dim() == 4:
        cos = cos[0]
        sin = sin[0]
    cos = cos.unsqueeze(unsqueeze_dim)
    sin = sin.unsqueeze(unsqueeze_dim)
    q_embed = (q * cos) + (rotate_half(q) * sin)
    k_embed = (k * cos) + (rotate_half(k) * sin)
    return q_embed, k_embed


class TalkerStackWrapper(nn.Module):
    """Runs the 28 decoder layers + final norm with tensor-only I/O."""

    def __init__(self, talker_model):
        super().__init__()
        self.layers = talker_model.layers
        self.norm = talker_model.norm

    def forward(self, hidden, cos, sin, mask):
        # position_embeddings tuple carries the reduced cos/sin; our patched
        # plain_rope consumes them. position_ids is unused by the attention math
        # (only cos/sin matter) so we pass None.
        pos_emb = (cos, sin)
        for layer in self.layers:
            hidden = layer(
                hidden,
                attention_mask=mask,
                position_ids=None,
                past_key_values=None,
                use_cache=False,
                cache_position=None,
                position_embeddings=pos_emb,
            )[0]
        return self.norm(hidden)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", default="tools/qwen3_tts/pnnx_out")
    ap.add_argument("--seq", type=int, default=8)
    ap.add_argument("--name", default="talker_decoder")
    ap.add_argument("--dynamic", action="store_true",
                    help="emit dynamic-seqlen graph (has non-native pnnx.Expression ops; "
                         "default static is clean and runtime-dynamic via Gemm 7=0)")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from qwen_tts import Qwen3TTSModel
    import qwen_tts.core.models.modeling_qwen3_tts as M

    def log(m):
        print(f"[convert_talker] {m}", flush=True)

    log(f"loading {args.model} (fp32, cpu) ...")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa"
    )
    talker = wrapper.model.talker
    cfg = talker.config
    H = cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", H // cfg.num_attention_heads)
    nlayers = cfg.num_hidden_layers
    log(f"hidden={H} head_dim={head_dim} layers={nlayers} rope_theta={cfg.rope_theta}")

    # --- monkeypatch mRoPE -> plain rotate_half (proven bit-exact) ---
    M.apply_multimodal_rotary_pos_emb = plain_rope
    log("patched apply_multimodal_rotary_pos_emb -> plain_rope")

    mod = TalkerStackWrapper(talker.model).eval()

    def make_inputs(S):
        hidden = torch.randn(1, S, H, dtype=torch.float32)
        pos = torch.arange(S).view(1, -1)  # (1,S) text positions
        pos3 = pos.unsqueeze(0).expand(3, -1, -1)  # (3,1,S) as rotary_emb expects
        with torch.no_grad():
            cos3, sin3 = talker.model.rotary_emb(hidden, pos3)  # (3,1,S,head_dim)
        cos = cos3[0].contiguous()  # (1,S,head_dim)
        sin = sin3[0].contiguous()
        mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1).view(1, 1, S, S)
        return hidden, cos, sin, mask

    S = args.seq
    hidden, cos, sin, mask = make_inputs(S)

    log("sanity forward ...")
    with torch.no_grad():
        y = mod(hidden, cos, sin, mask)
    log(f"forward OK, output shape={tuple(y.shape)}")

    import pnnx
    ptpath = os.path.join(args.outdir, f"{args.name}.pt")
    export_kwargs = dict(
        inputs=(hidden, cos, sin, mask),
        input_shapes=[[1, S, H], [1, S, head_dim], [1, S, head_dim], [1, 1, S, S]],
        input_types=["f32", "f32", "f32", "f32"],
        ncnnparam=os.path.join(args.outdir, f"{args.name}.ncnn.param"),
        ncnnbin=os.path.join(args.outdir, f"{args.name}.ncnn.bin"),
        fp16=False,
        optlevel=2,
    )
    if args.dynamic:
        # Second input set at a DIFFERENT seqlen marks the seq dim dynamic (-1).
        # WARNING: this makes pnnx emit pnnx.Expression + Tensor.expand ops (from
        # the GQA repeat_kv shape arithmetic) that plain ncnn cannot execute. The
        # STATIC export (default) emits clean native Tile/ExpandDims/Reshape (like
        # the shipped qwen3 decoder) and ncnn Gemm 7=0 handles variable M at
        # RUNTIME anyway — so static is both clean AND seqlen-flexible at run time.
        S2 = S * 2 + 1
        hidden2, cos2, sin2, mask2 = make_inputs(S2)
        export_kwargs.update(
            inputs2=(hidden2, cos2, sin2, mask2),
            input_shapes2=[[1, S2, H], [1, S2, head_dim], [1, S2, head_dim], [1, 1, S2, S2]],
            input_types2=["f32", "f32", "f32", "f32"],
        )
        log(f"pnnx.export (DYNAMIC seqlen via inputs2 S2={S2}) ...")
    else:
        log(f"pnnx.export (STATIC trace seq={S}, runtime-dynamic via Gemm 7=0) ...")
    pnnx.export(mod, ptpath, **export_kwargs)
    log("pnnx.export returned")

    # --- post-process the .param ---
    # For the clean STATIC export, first make the seqlen dynamic (rewrite baked
    # trace-seqlen in q/k/v/o Reshape ops to -1, matching the shipped decoder),
    # THEN patch in the SDPA kv_cache. The dynamic export path is already
    # seqlen-flexible (but has non-native ops) so it skips dynamize.
    from add_kvcache import add_kvcache_to_param
    base = os.path.join(args.outdir, f"{args.name}.ncnn.param")

    if not args.dynamic:
        from dynamize_seqlen import dynamize
        dyn = os.path.join(args.outdir, f"{args.name}_dyn.ncnn.param")
        nrw = dynamize(base, dyn, S)
        log(f"dynamized seqlen: {nrw} reshape rewrites -> {dyn}")
        src = dyn
    else:
        src = base

    dst = os.path.join(args.outdir, f"{args.name}_kvcache.ncnn.param")
    n = add_kvcache_to_param(src, dst)
    log(f"kv_cache patched: {n} SDPA layers -> {dst}")
    log(f"FINAL AR-ready param: {dst} (bin: {args.name}.ncnn.bin)")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()
