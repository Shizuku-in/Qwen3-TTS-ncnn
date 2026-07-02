# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# Convert the 5-layer Qwen3-TTS Code Predictor (sub-talker / MTP) backbone to
# ncnn via pnnx.
#
# Structurally identical to the Talker backbone (RMSNorm/Gemm/RotaryEmbed/SDPA/
# SwiGLU, GQA 16Q/8KV, head_dim 128, hidden 1024, RoPE theta 1e6) EXCEPT:
#   - 5 layers instead of 28
#   - uses the standard apply_rotary_pos_emb (plain rotate_half, 1-D RoPE) rather
#     than interleaved mRoPE, so NO monkeypatch is needed. We feed a single
#     (1,S,head_dim) cos/sin table directly.
#
# Reuses the same wrapper + dynamize-seqlen + add_kvcache post-processing proven
# on the Talker (see convert_talker.py). The Code Predictor runs its own fresh
# KV cache each frame across the 15 inner passes.
#
# Usage:
#   python convert_code_predictor.py --model models/Qwen3-TTS-12Hz-0.6B-Base \
#       --outdir tools/qwen3_tts/pnnx_out --seq 8

import argparse
import os
import sys
import torch
import torch.nn as nn


class BackboneWrapper(nn.Module):
    """Runs the decoder layers + final norm with tensor-only I/O.

    Matches the shipped decoder contract: in0=inputs_embeds (H x S), in1=cos
    (1,S,head_dim), in2=sin, in3=additive causal mask (1,1,S,S).
    """

    def __init__(self, backbone):
        super().__init__()
        self.layers = backbone.layers
        self.norm = backbone.norm

    def forward(self, hidden, cos, sin, mask):
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
    ap.add_argument("--name", default="code_predictor")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from qwen_tts import Qwen3TTSModel

    def log(m):
        print(f"[convert_cp] {m}", flush=True)

    log(f"loading {args.model} (fp32, cpu) ...")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa"
    )
    talker = wrapper.model.talker
    cp = talker.code_predictor
    backbone = cp.model  # Qwen3TTSTalkerCodePredictorModel
    cfg = cp.config
    H = cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", H // cfg.num_attention_heads)
    nlayers = cfg.num_hidden_layers
    log(f"hidden={H} head_dim={head_dim} layers={nlayers} rope_theta={cfg.rope_theta}")

    mod = BackboneWrapper(backbone).eval()

    def make_inputs(S):
        hidden = torch.randn(1, S, H, dtype=torch.float32)
        pos = torch.arange(S).view(1, -1)
        with torch.no_grad():
            cos, sin = backbone.rotary_emb(hidden, pos)  # (1,S,head_dim)
        cos = cos.contiguous()
        sin = sin.contiguous()
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
    log(f"pnnx.export (STATIC trace seq={S}) ...")
    pnnx.export(
        mod,
        ptpath,
        inputs=(hidden, cos, sin, mask),
        input_shapes=[[1, S, H], [1, S, head_dim], [1, S, head_dim], [1, 1, S, S]],
        input_types=["f32", "f32", "f32", "f32"],
        ncnnparam=os.path.join(args.outdir, f"{args.name}.ncnn.param"),
        ncnnbin=os.path.join(args.outdir, f"{args.name}.ncnn.bin"),
        fp16=False,
        optlevel=2,
    )
    log("pnnx.export returned")

    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    from dynamize_seqlen import dynamize
    from add_kvcache import add_kvcache_to_param

    src = os.path.join(args.outdir, f"{args.name}.ncnn.param")
    dyn = os.path.join(args.outdir, f"{args.name}_dyn.ncnn.param")
    dynamize(src, dyn, S)
    log(f"dynamized seqlen -> {dyn}")

    dst = os.path.join(args.outdir, f"{args.name}_kvcache.ncnn.param")
    n = add_kvcache_to_param(dyn, dst)
    log(f"kv_cache patched: {n} SDPA layers -> {dst}")
    log(f"FINAL AR-ready param: {dst} (bin: {args.name}.ncnn.bin)")


if __name__ == "__main__":
    main()
