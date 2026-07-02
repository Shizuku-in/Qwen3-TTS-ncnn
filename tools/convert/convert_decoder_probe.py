# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# P4 probe: trace the Qwen3-TTS 12Hz speech-decoder post-RVQ stack through pnnx
# to discover which custom ops it can lower and which need hand-written ncnn
# layers.
#
# The decoder forward is:
#     hidden = quantizer.decode(codes)          # RVQ dequant -> (B, 512, T)  [host-side, NOT traced]
#     hidden = pre_conv(hidden).transpose(1,2)  # CausalConv 512->1024 -> (B,T,1024)
#     hidden = pre_transformer(hidden)          # 8 sliding-attn layers -> (B,T,1024)
#     hidden = hidden.permute(0,2,1)            # (B,1024,T)
#     for blocks in upsample: ...               # 2x CausalTransConv + ConvNeXt -> (B,1024,4T)
#     for block in decoder: ...                 # CausalConv + 4 DecoderBlock + SnakeBeta + CausalConv -> (B,1,1920T)
#     wav = clamp(-1,1)
#
# We trace from the DEQUANTIZED (B,512,T) tensor, skipping the integer-gather RVQ
# (which is done host-side from precomputed tables). This isolates the float
# graph so we can see pnnx's handling of CausalConv dynamic pad, SnakeBeta,
# CausalTransConv trim, and the sliding-window attention.
#
# Usage:
#   python convert_decoder_probe.py --model models/Qwen3-TTS-12Hz-0.6B-Base \
#       --outdir tools/qwen3_tts/pnnx_out --frames 8

import argparse
import os
import torch
import torch.nn as nn


class DecoderStackWrapper(nn.Module):
    """Runs the decoder from the dequantized (B,512,T) latent to waveform."""

    def __init__(self, decoder):
        super().__init__()
        self.pre_conv = decoder.pre_conv
        self.pre_transformer = decoder.pre_transformer
        self.upsample = decoder.upsample
        self.decoder = decoder.decoder

    def forward(self, hidden):  # hidden: (B, 512, T)
        hidden = self.pre_conv(hidden).transpose(1, 2)
        hidden = self.pre_transformer(inputs_embeds=hidden).last_hidden_state
        hidden = hidden.permute(0, 2, 1)
        for blocks in self.upsample:
            for block in blocks:
                hidden = block(hidden)
        wav = hidden
        for block in self.decoder:
            wav = block(wav)
        return wav.clamp(min=-1, max=1)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", default="tools/qwen3_tts/pnnx_out")
    ap.add_argument("--frames", type=int, default=8)
    ap.add_argument("--name", default="decoder_stack")
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    from qwen_tts import Qwen3TTSModel

    def log(m):
        print(f"[decoder_probe] {m}", flush=True)

    log(f"loading {args.model} (fp32, cpu) ...")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa"
    )
    st = wrapper.model.speech_tokenizer
    decoder = st.model.decoder
    dcfg = decoder.config
    codebook_dim = dcfg.codebook_dim  # 512
    log(f"codebook_dim={codebook_dim} latent={dcfg.latent_dim} decoder_dim={dcfg.decoder_dim} "
        f"layers={dcfg.num_hidden_layers} upsample_rates={dcfg.upsample_rates} "
        f"upsampling_ratios={dcfg.upsampling_ratios}")

    mod = DecoderStackWrapper(decoder).eval()

    T = args.frames
    hidden = torch.randn(1, codebook_dim, T, dtype=torch.float32)

    log("sanity forward ...")
    with torch.no_grad():
        y = mod(hidden)
    log(f"forward OK, output shape={tuple(y.shape)}  (expect ~(1,1,{T*1920}))")

    import pnnx
    ptpath = os.path.join(args.outdir, f"{args.name}.pt")
    log("pnnx.export (static trace) ...")
    pnnx.export(
        mod,
        ptpath,
        inputs=(hidden,),
        input_shapes=[[1, codebook_dim, T]],
        input_types=["f32"],
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
