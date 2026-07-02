# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# Convert the Qwen3-TTS 12Hz speech-decoder CONV STACK to ncnn via pnnx.
#
# The decoder forward (modeling_qwen3_tts_tokenizer_v2.py L869) is:
#   quantizer.decode(codes)        -> (B,512,T)   [RVQ dequant, host-side]
#   pre_conv (512->1024, causal k3)-> (B,1024,T)
#   transpose -> pre_transformer   -> (B,T,1024)  [DONE: convert_decoder_transformer.py]
#   permute -> (B,1024,T)
#   upsample ModuleList (2x)        -> (B,1024,4T)  [CausalTransConv + ConvNeXt]
#   decoder ModuleList              -> (B,1,1920*..)[CausalConv + 4 DecoderBlock + SnakeBeta + CausalConv]
#   clamp(-1,1)
#
# This script converts the parts made of custom conv/activation ops:
#   part A: pre_conv           (512 -> 1024)
#   part B: upsample + decoder (1024 -> 1 waveform), the SnakeBeta/CausalConv/
#           CausalTransConv-heavy tail.
# We probe whether pnnx can lower SnakeBeta (sin/pow/exp elementwise),
# CausalConvNet (static left-pad + Conv1d at stride 1), and CausalTransConvNet
# (ConvTranspose1d + fixed right-trim) automatically. Whatever does not lower
# is reported so we know which custom ncnn::Layer to hand-write.
#
# Usage:
#   python convert_decoder_convstack.py --model models/Qwen3-TTS-12Hz-0.6B-Base \
#       --outdir tools/qwen3_tts/pnnx_out --frames 8

import argparse
import os
import torch
import torch.nn as nn


class PreConvWrapper(nn.Module):
    def __init__(self, decoder):
        super().__init__()
        self.pre_conv = decoder.pre_conv

    def forward(self, x):  # x: (B,512,T)
        return self.pre_conv(x)  # (B,1024,T)


class ConvTailWrapper(nn.Module):
    """upsample (2x ConvNeXt) + decoder ModuleList -> waveform, minus final clamp
    (clamp is trivial to do host-side / as a ncnn Clip)."""

    def __init__(self, decoder):
        super().__init__()
        self.upsample = decoder.upsample
        self.decoder = decoder.decoder

    def forward(self, x):  # x: (B,1024,T)
        for blocks in self.upsample:
            for block in blocks:
                x = block(x)
        for block in self.decoder:
            x = block(x)
        return x  # (B,1,1920*4*T-ish)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", default="tools/qwen3_tts/pnnx_out")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--part", choices=["pre", "tail", "both"], default="both")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from qwen_tts import Qwen3TTSModel

    def log(m):
        print(f"[convert_convstack] {m}", flush=True)

    log(f"loading {args.model} (fp32, cpu) ...")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa"
    )
    decoder = wrapper.model.speech_tokenizer.model.decoder
    dcfg = decoder.config
    latent = dcfg.latent_dim          # 1024
    codebook_dim = dcfg.codebook_dim  # 512
    log(f"latent={latent} codebook_dim={codebook_dim} decoder_dim={dcfg.decoder_dim} "
        f"upsample_rates={dcfg.upsample_rates} upsampling_ratios={dcfg.upsampling_ratios}")

    import pnnx
    T = args.frames

    if args.part in ("pre", "both"):
        mod = PreConvWrapper(decoder).eval()
        x = torch.randn(1, codebook_dim, T, dtype=torch.float32)
        with torch.no_grad():
            y = mod(x)
        log(f"pre_conv forward OK: {tuple(x.shape)} -> {tuple(y.shape)}")
        pnnx.export(
            mod, os.path.join(args.outdir, "dec_pre_conv.pt"),
            inputs=(x,), input_shapes=[[1, codebook_dim, T]], input_types=["f32"],
            ncnnparam=os.path.join(args.outdir, "dec_pre_conv.ncnn.param"),
            ncnnbin=os.path.join(args.outdir, "dec_pre_conv.ncnn.bin"),
            fp16=False, optlevel=2,
        )
        log("pre_conv exported")

    if args.part in ("tail", "both"):
        mod = ConvTailWrapper(decoder).eval()
        x = torch.randn(1, latent, T, dtype=torch.float32)
        with torch.no_grad():
            y = mod(x)
        log(f"conv tail forward OK: {tuple(x.shape)} -> {tuple(y.shape)} (expect ~{T*1920})")
        pnnx.export(
            mod, os.path.join(args.outdir, "dec_conv_tail.pt"),
            inputs=(x,), input_shapes=[[1, latent, T]], input_types=["f32"],
            ncnnparam=os.path.join(args.outdir, "dec_conv_tail.ncnn.param"),
            ncnnbin=os.path.join(args.outdir, "dec_conv_tail.ncnn.bin"),
            fp16=False, optlevel=2,
        )
        log("conv tail exported")


if __name__ == "__main__":
    main()
