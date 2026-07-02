# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# pnnx conversion PROOF-OF-CONCEPT for the Qwen3-TTS Talker (issue #6791).
#
# Goal: de-risk the crux of the whole port — can pnnx convert ONE Talker decoder
# layer (Qwen3 attention: q/k/v proj, per-head q/k RMSNorm, interleaved mRoPE,
# GQA 16Q/8KV, SDPA, SwiGLU MLP) to ncnn at all?
#
# Strategy (from NOTES_talker_forward_map.md): pnnx traces torchscript, so we
# must trace a TENSOR-IN/TENSOR-OUT module — never the HF generate loop. We wrap
# a single real decoder layer and feed it explicit (hidden, cos, sin, mask). The
# KV cache is NOT traced; it is patched into the .param post-hoc (separate script).
#
# This uses REAL weights from the checkpoint so the traced graph is faithful.
#
# Usage:
#   .venv-tts/bin/python tools/qwen3_tts/scripts/convert_talker_poc.py \
#       --model models/Qwen3-TTS-12Hz-0.6B-Base --out tools/qwen3_tts/pnnx_out

import argparse
import os

import torch
import torch.nn as nn


def log(*a):
    print("[convert_poc]", *a, flush=True)


class OneLayerWrapper(nn.Module):
    """Tensor-in/tensor-out wrapper around one real Talker decoder layer.

    We bypass the HF generate machinery: cos/sin (already mRoPE-reduced to a
    single per-position table on the host) and an additive attention mask are
    passed as plain tensors. No KV cache in the traced graph.
    """

    def __init__(self, layer):
        super().__init__()
        self.layer = layer

    def forward(self, hidden_states, cos, sin, attention_mask):
        # matches Qwen3TTSTalkerDecoderLayer.forward signature (tensor subset)
        out = self.layer(
            hidden_states=hidden_states,
            attention_mask=attention_mask,
            position_embeddings=(cos, sin),
            use_cache=False,
        )
        # layer returns a tuple (hidden, attn_weights|None)
        return out[0] if isinstance(out, tuple) else out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--out", default="tools/qwen3_tts/pnnx_out")
    ap.add_argument("--seq", type=int, default=4, help="trace sequence length")
    args = ap.parse_args()
    os.makedirs(args.out, exist_ok=True)

    from qwen_tts import Qwen3TTSModel

    log(f"loading {args.model} (fp32, cpu) ...")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa",
    )
    talker = wrapper.model.talker
    cfg = talker.config
    layer = talker.model.layers[0].eval()

    H = cfg.hidden_size            # 1024
    n_heads = cfg.num_attention_heads   # 16
    head_dim = getattr(cfg, "head_dim", H // n_heads)  # 128
    log(f"hidden={H} heads={n_heads} head_dim={head_dim} "
        f"kv_heads={cfg.num_key_value_heads} rope_theta={cfg.rope_theta}")

    S = args.seq
    torch.manual_seed(0)
    hidden = torch.randn(1, S, H, dtype=torch.float32)
    # mRoPE cos/sin arrive as (3, B, S, head_dim): 3 sections (temporal/height/width).
    # In the TTS text path all 3 rows are equal -> apply_interleaved_rope collapses to
    # row 0. We build 3 equal rows so the real layer runs faithfully; the C++ port will
    # precompute the reduced single-table form on the host.
    base_cos = torch.randn(1, S, head_dim, dtype=torch.float32)
    base_sin = torch.randn(1, S, head_dim, dtype=torch.float32)
    cos = base_cos.unsqueeze(0).expand(3, -1, -1, -1).contiguous()  # (3,1,S,128)
    sin = base_sin.unsqueeze(0).expand(3, -1, -1, -1).contiguous()
    # additive causal mask (1,1,S,S): 0 on/below diag, -inf above
    mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1).view(1, 1, S, S)

    mod = OneLayerWrapper(layer).eval()

    log("sanity forward ...")
    with torch.no_grad():
        y = mod(hidden, cos, sin, mask)
    log(f"forward OK, output shape={tuple(y.shape)}")

    # --- pnnx export (traces internally, emits pnnx + ncnn artifacts) ---
    pt_path = os.path.join(args.out, "talker_layer0.pt")
    shapes = [[1, S, H], [3, 1, S, head_dim], [3, 1, S, head_dim], [1, 1, S, S]]
    types = ["f32", "f32", "f32", "f32"]
    log(f"pnnx.export (input_shapes={shapes}) ...")
    import pnnx
    pnnx.export(
        mod,
        pt_path,
        inputs=(hidden, cos, sin, mask),
        input_shapes=shapes,
        input_types=types,
        check_trace=False,
        fp16=False,  # keep fp32 for faithful parity comparison
        ncnnparam=os.path.join(args.out, "talker_layer0.ncnn.param"),
        ncnnbin=os.path.join(args.out, "talker_layer0.ncnn.bin"),
    )
    log("pnnx.export returned")

    # report artifacts
    for f in sorted(os.listdir(args.out)):
        log("  artifact:", f)


if __name__ == "__main__":
    main()
