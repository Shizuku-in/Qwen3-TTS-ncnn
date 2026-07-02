# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# Numerical parity: converted ncnn speech-decoder transformer vs PyTorch.
#
# Mirrors convert_decoder_transformer.py's wrapper (bypass HF create_causal_mask,
# feed explicit sliding-window additive mask + reduced cos/sin), runs both the
# torch reference and the ncnn graph on identical input, reports cosine sim.

import argparse
import os
import numpy as np
import torch
import torch.nn as nn


def rotate_half(x):
    x1 = x[..., : x.shape[-1] // 2]
    x2 = x[..., x.shape[-1] // 2 :]
    return torch.cat((-x2, x1), dim=-1)


def apply_rope(q, k, cos, sin):
    cos = cos.unsqueeze(1)
    sin = sin.unsqueeze(1)
    return (q * cos) + (rotate_half(q) * sin), (k * cos) + (rotate_half(k) * sin)


class TFWrapper(nn.Module):
    """8-layer decoder transformer with tensor-only I/O (input_proj/output_proj inside)."""

    def __init__(self, tf):
        super().__init__()
        self.tf = tf

    def forward(self, hidden, cos, sin, mask):
        m = self.tf
        h = m.input_proj(hidden)
        pos_emb = (cos, sin)
        for layer in m.layers:
            residual = h
            x = layer.input_layernorm(h)
            x, _ = layer.self_attn(hidden_states=x, position_embeddings=pos_emb,
                                   attention_mask=mask, past_key_values=None)
            h = residual + layer.self_attn_layer_scale(x)
            residual = h
            x = layer.post_attention_layernorm(h)
            x = layer.mlp(x)
            h = residual + layer.mlp_layer_scale(x)
        h = m.norm(h)
        return m.output_proj(h)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--param", default="tools/qwen3_tts/pnnx_out/dec_transformer.ncnn.param")
    ap.add_argument("--bin", default="tools/qwen3_tts/pnnx_out/dec_transformer.ncnn.bin")
    ap.add_argument("--frames", type=int, default=16)
    args = ap.parse_args()

    def log(m):
        print(f"[parity_dtf] {m}", flush=True)

    from qwen_tts import Qwen3TTSModel
    log("loading (fp32, cpu) ...")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa"
    )
    tf = wrapper.model.speech_tokenizer.model.decoder.pre_transformer.eval()
    cfg = tf.config
    latent = cfg.latent_dim
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    window = cfg.sliding_window
    T = args.frames

    torch.manual_seed(0)
    hidden = torch.randn(1, T, latent, dtype=torch.float32)
    pos = torch.arange(T).view(1, -1)
    with torch.no_grad():
        cos, sin = tf.rotary_emb(hidden, pos)

    # sliding-window causal additive mask (window 72)
    mask = torch.full((T, T), float("-inf"))
    for i in range(T):
        lo = max(0, i - window + 1)
        mask[i, lo : i + 1] = 0.0
    mask = mask.view(1, 1, T, T)

    mod = TFWrapper(tf).eval()
    log(f"torch ref (T={T}) ...")
    with torch.no_grad():
        y_ref = mod(hidden, cos, sin, mask).squeeze(0).numpy()
    log(f"torch out {y_ref.shape}")

    import ncnn
    log("ncnn ...")
    with ncnn.Net() as net:
        net.load_param(args.param)
        net.load_model(args.bin)
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(hidden.squeeze(0).numpy()).clone())
            ex.input("in1", ncnn.Mat(cos.squeeze(0).numpy()).clone())
            ex.input("in2", ncnn.Mat(sin.squeeze(0).numpy()).clone())
            ex.input("in3", ncnn.Mat(mask.squeeze(0).numpy()).clone())
            _, out0 = ex.extract("out0")
            y_ncnn = np.array(out0)

    log(f"shapes ref={y_ref.shape} ncnn={y_ncnn.shape}")
    diff = np.abs(y_ref - y_ncnn)
    a = y_ref.reshape(-1)
    b = y_ncnn.reshape(-1)
    cos_sim = float(a @ b / (np.linalg.norm(a) * np.linalg.norm(b) + 1e-12))
    log(f"max_abs_diff={diff.max():.3e} mean={diff.mean():.3e} cos_sim={cos_sim:.6f}")
    log(f"VERDICT: {'PASS' if cos_sim > 0.9999 else 'FAIL'}")


if __name__ == "__main__":
    main()
