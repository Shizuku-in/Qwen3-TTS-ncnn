# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# P2 numerical parity gate: converted ncnn Talker backbone vs PyTorch reference.
#
# Runs the same fixed input through:
#   (a) the PyTorch TalkerStackWrapper (28 layers + final norm), with mRoPE
#       monkeypatched to plain_rope (proven bit-exact by verify_mrope_collapse.py)
#   (b) the pnnx-converted ncnn talker_decoder (prefill, no kv_cache)
# and reports max abs / rel difference on the output hidden states.
#
# This validates the whole conversion recipe end-to-end (weights, layout, RoPE,
# GQA, SDPA) before we build the C++ AR loop against it.
#
# Usage:
#   python parity_talker.py --model models/Qwen3-TTS-12Hz-0.6B-Base \
#       --param tools/qwen3_tts/pnnx_out/talker_decoder.ncnn.param \
#       --bin   tools/qwen3_tts/pnnx_out/talker_decoder.ncnn.bin --seq 8

import argparse
import os
import numpy as np
import torch

# reuse the proven wrapper + plain_rope from the converter
import sys
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
from convert_talker import TalkerStackWrapper, plain_rope


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--param", required=True)
    ap.add_argument("--bin", required=True)
    ap.add_argument("--seq", type=int, default=8)
    args = ap.parse_args()

    def log(m):
        print(f"[parity] {m}", flush=True)

    from qwen_tts import Qwen3TTSModel
    import qwen_tts.core.models.modeling_qwen3_tts as M

    log(f"loading {args.model} (fp32, cpu) ...")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa"
    )
    talker = wrapper.model.talker
    cfg = talker.config
    H = cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", H // cfg.num_attention_heads)
    S = args.seq

    # patch mRoPE -> plain_rope (same as conversion)
    M.apply_multimodal_rotary_pos_emb = plain_rope
    mod = TalkerStackWrapper(talker.model).eval()

    # fixed deterministic input
    torch.manual_seed(1234)
    hidden = torch.randn(1, S, H, dtype=torch.float32)
    pos3 = torch.arange(S).view(1, -1).unsqueeze(0).expand(3, -1, -1)
    with torch.no_grad():
        cos3, sin3 = talker.model.rotary_emb(hidden, pos3)
    cos = cos3[0].contiguous()
    sin = sin3[0].contiguous()
    mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1).view(1, 1, S, S)

    log("running PyTorch reference ...")
    with torch.no_grad():
        y_ref = mod(hidden, cos, sin, mask).squeeze(0).numpy()  # (S, H)

    log("running ncnn ...")
    import ncnn
    with ncnn.Net() as net:
        net.load_param(args.param)
        net.load_model(args.bin)
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(hidden.squeeze(0).numpy()).clone())
            ex.input("in1", ncnn.Mat(cos.squeeze(0).numpy()).clone())
            ex.input("in2", ncnn.Mat(sin.squeeze(0).numpy()).clone())
            ex.input("in3", ncnn.Mat(mask.squeeze(0).numpy()).clone())
            _, out0 = ex.extract("out0")
            y_ncnn = np.array(out0)  # (S, H)

    log(f"y_ref shape={y_ref.shape} y_ncnn shape={y_ncnn.shape}")
    if y_ref.shape != y_ncnn.shape:
        log(f"SHAPE MISMATCH")
        return

    diff = np.abs(y_ref - y_ncnn)
    denom = np.maximum(np.abs(y_ref), 1e-6)
    rel = diff / denom
    log(f"max_abs_diff = {diff.max():.6e}")
    log(f"mean_abs_diff = {diff.mean():.6e}")
    log(f"max_rel_diff = {rel.max():.6e}")
    # per-position (row) max abs diff — spikes at one row => masking/pos bug;
    # steady growth => benign fp32 accumulation over 28 layers
    per_row = diff.max(axis=1)
    log("per-position max_abs_diff: " + " ".join(f"{v:.2e}" for v in per_row))
    # cosine similarity per row (robust to scale) — should be ~1.0
    cos_sim = [float(np.dot(y_ref[i], y_ncnn[i]) /
                     (np.linalg.norm(y_ref[i]) * np.linalg.norm(y_ncnn[i]) + 1e-9))
               for i in range(y_ref.shape[0])]
    log("per-position cosine_sim: " + " ".join(f"{v:.6f}" for v in cos_sim))
    # correlation over all elements
    corr = float(np.corrcoef(y_ref.ravel(), y_ncnn.ravel())[0, 1])
    log(f"overall pearson corr = {corr:.8f}")
    # fp32 through 28 layers: judge by cosine sim, not raw abs
    ok = min(cos_sim) > 0.9999
    log(f"VERDICT: {'PASS' if ok else 'FAIL'} (min cosine_sim > 0.9999)")


if __name__ == "__main__":
    main()
