# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# Convert the 12Hz speech-decoder's pre_transformer (8-layer sliding-window
# attention) to ncnn via pnnx.
#
# The decoder's forward() traces cleanly for the conv parts but dies in the
# transformer's create_causal_mask -> vmap (same HF-mask issue as the Talker).
# Fix: wrap the transformer layer loop with tensor-only I/O, feeding an explicit
# additive sliding-window causal mask + precomputed cos/sin (standard RoPE,
# theta=10000). Attention here uses plain apply_rotary_pos_emb (no mRoPE), no
# GQA (16q=16kv), no QK-norm (q_norm/k_norm=Identity) -> clean graph expected.
#
# Includes input_proj (latent 1024 -> hidden 512) and output_proj (512 -> 1024)
# so the exported net maps (T,1024) -> (T,1024), matching the decoder's
# pre_transformer(inputs_embeds=hidden).last_hidden_state contract.
#
# Usage:
#   python convert_decoder_transformer.py --model models/Qwen3-TTS-12Hz-0.6B-Base

import argparse
import os
import torch
import torch.nn as nn


class DecoderTransformerWrapper(nn.Module):
    """input_proj -> 8 sliding-window layers -> norm -> output_proj, tensor I/O."""

    def __init__(self, tf):
        super().__init__()
        self.input_proj = tf.input_proj
        self.layers = tf.layers
        self.norm = tf.norm
        self.output_proj = tf.output_proj

    def forward(self, hidden, cos, sin, mask):
        # hidden: (1, T, latent_dim=1024)
        hidden = self.input_proj(hidden)  # -> (1, T, 512)
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
            )
        hidden = self.norm(hidden)
        hidden = self.output_proj(hidden)  # -> (1, T, 1024)
        return hidden


def build_sliding_mask(T, window):
    # additive mask (1,1,T,T): 0 where attendable, -inf otherwise.
    # causal + sliding window: key j visible to query i iff  i-window < j <= i
    m = torch.full((T, T), float("-inf"))
    for i in range(T):
        lo = max(0, i - window + 1)
        m[i, lo : i + 1] = 0.0
    return m.view(1, 1, T, T)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", default="tools/qwen3_tts/pnnx_out")
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--name", default="decoder_transformer")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from qwen_tts import Qwen3TTSModel
    import qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2 as V2

    def log(m):
        print(f"[convert_dec_tf] {m}", flush=True)

    log(f"loading {args.model} (fp32, cpu) ...")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa"
    )
    tf = wrapper.model.speech_tokenizer.model.decoder.pre_transformer
    cfg = tf.config
    latent = cfg.latent_dim
    hidden_size = cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", hidden_size // cfg.num_attention_heads)
    window = cfg.sliding_window
    theta = cfg.rope_theta
    log(f"latent={latent} hidden={hidden_size} head_dim={head_dim} layers={cfg.num_hidden_layers} "
        f"window={window} theta={theta}")

    mod = DecoderTransformerWrapper(tf).eval()

    T = args.frames

    def make_inputs(TT):
        hidden = torch.randn(1, TT, latent, dtype=torch.float32)
        pos = torch.arange(TT).view(1, -1)
        with torch.no_grad():
            cos, sin = tf.rotary_emb(hidden, pos)  # (1,TT,head_dim)
        cos = cos.contiguous()
        sin = sin.contiguous()
        mask = build_sliding_mask(TT, window)
        return hidden, cos, sin, mask

    hidden, cos, sin, mask = make_inputs(T)

    log(f"cos shape={tuple(cos.shape)} mask shape={tuple(mask.shape)}")
    log("sanity forward ...")
    with torch.no_grad():
        y = mod(hidden, cos, sin, mask)
    log(f"forward OK, output shape={tuple(y.shape)}")

    import pnnx
    ptpath = os.path.join(args.outdir, f"{args.name}.pt")
    log("pnnx.export (static trace) ...")
    pnnx.export(
        mod,
        ptpath,
        inputs=(hidden, cos, sin, mask),
        input_shapes=[[1, T, latent], [1, T, head_dim], [1, T, head_dim], [1, 1, T, T]],
        input_types=["f32", "f32", "f32", "f32"],
        ncnnparam=os.path.join(args.outdir, f"{args.name}.ncnn.param"),
        ncnnbin=os.path.join(args.outdir, f"{args.name}.ncnn.bin"),
        fp16=False,
        optlevel=2,
    )
    log("pnnx.export returned")


if __name__ == "__main__":
    import sys
    sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))
    main()


def parity_check(model_path, param, binf, T=16):
    """Standalone parity: torch decoder-transformer vs ncnn, on identical input."""
    import numpy as np
    import torch
    from qwen_tts import Qwen3TTSModel

    w = Qwen3TTSModel.from_pretrained(model_path, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa")
    tf = w.model.speech_tokenizer.model.decoder.pre_transformer.eval()
    cfg = tf.config
    latent = cfg.latent_dim
    head_dim = getattr(cfg, "head_dim", cfg.hidden_size // cfg.num_attention_heads)
    win = cfg.sliding_window

    torch.manual_seed(1)
    hidden = torch.randn(1, T, latent, dtype=torch.float32)
    pos = torch.arange(T).view(1, -1)
    with torch.no_grad():
        cos, sin = tf.rotary_emb(hidden, pos)
    # sliding-window causal additive mask
    m = torch.full((T, T), float("-inf"))
    for i in range(T):
        lo = max(0, i - win + 1)
        m[i, lo:i+1] = 0.0
    mask = m.view(1, 1, T, T)

    with torch.no_grad():
        y_ref = tf(inputs_embeds=hidden).last_hidden_state.squeeze(0).numpy()

    import ncnn
    with ncnn.Net() as net:
        net.load_param(param); net.load_model(binf)
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(hidden.squeeze(0).numpy()).clone())
            ex.input("in1", ncnn.Mat(cos.squeeze(0).numpy()).clone())
            ex.input("in2", ncnn.Mat(sin.squeeze(0).numpy()).clone())
            ex.input("in3", ncnn.Mat(mask.squeeze(0).numpy()).clone())
            _, out0 = ex.extract("out0")
            y = np.array(out0)

    diff = np.abs(y_ref - y)
    a = y_ref.flatten(); b = y.flatten()
    cos_sim = float(np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)))
    print(f"[parity_dec_tf] shapes ref={y_ref.shape} ncnn={y.shape}")
    print(f"[parity_dec_tf] max_abs_diff={diff.max():.3e} mean={diff.mean():.3e} cos_sim={cos_sim:.6f}")
    print(f"[parity_dec_tf] VERDICT: {'PASS' if cos_sim > 0.9999 else 'FAIL'}")


if __name__ == "__main__" and len(__import__('sys').argv) > 1 and __import__('sys').argv[1] == "parity":
    import sys
    parity_check(sys.argv[2], sys.argv[3], sys.argv[4])
    sys.exit(0)
