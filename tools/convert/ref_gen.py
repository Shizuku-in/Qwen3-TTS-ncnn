# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# P0 reference-generation gate for the Qwen3-TTS -> ncnn port (issue #6791).
#
# Runs the PyTorch reference (0.6B-Base, voice-clone) under DETERMINISTIC
# (greedy) decoding and dumps the golden artifacts that every later ncnn stage
# is validated against:
#   - talker_codes  [T, 16]  int64   -> dumps/golden_codes.npy + .txt
#   - output wav    float32          -> dumps/golden_out.wav
#
# Determinism: do_sample=False + subtalker_dosample=False + fixed seed. This is
# the parity contract agreed for the port (compare pre-sampling argmax / codes,
# not bit-exact audio through stochastic sampling).
#
# Usage:
#   .venv-tts/bin/python tools/qwen3_tts/scripts/ref_gen.py \
#       --model models/Qwen3-TTS-12Hz-0.6B-Base \
#       --ref   tools/qwen3_tts/dumps/ref_input.wav \
#       --text  "这是一个用于验证 ncnn 移植数值一致性的测试句子。" \
#       --language Chinese

import argparse
import os

import numpy as np
import soundfile as sf
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True, help="path to Qwen3-TTS-12Hz-0.6B-Base")
    ap.add_argument("--ref", required=True, help="reference wav (voice-clone prompt)")
    ap.add_argument("--text", default="这是一个用于验证 ncnn 移植数值一致性的测试句子。")
    ap.add_argument("--language", default="Chinese")
    ap.add_argument("--outdir", default="tools/qwen3_tts/dumps")
    ap.add_argument("--max-new-tokens", type=int, default=256)
    args = ap.parse_args()

    os.makedirs(args.outdir, exist_ok=True)

    # ---- determinism ----
    # Greedy decoding (do_sample=False) is deterministic at the sampling level.
    # NOTE: do NOT enable torch.use_deterministic_algorithms here — it routes
    # replication_pad through a buggy decomposition on torch 2.12
    # (_unsafe_index found unexpected index type Float) in the Mimi encoder.
    torch.manual_seed(0)
    np.random.seed(0)

    from qwen_tts import Qwen3TTSModel

    print(f"[ref_gen] loading {args.model} ...", flush=True)
    model = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map="cuda:0",
        dtype=torch.float32,  # fp32 golden reference; also avoids bf16 replication-pad decomp bug in encoder
        attn_implementation="sdpa",  # match pnnx conversion path (SDPA), not flash-attn
    )
    print("[ref_gen] model loaded", flush=True)

    wav, sr = sf.read(args.ref, dtype="float32", always_2d=False)
    if wav.ndim > 1:
        wav = wav.mean(axis=-1).astype(np.float32)

    # x_vector_only_mode=True -> speaker embedding only, no ref_text needed.
    prompt = model.create_voice_clone_prompt(
        ref_audio=(wav, sr),
        ref_text=None,
        x_vector_only_mode=True,
    )

    # Hook: capture the golden talker_codes [T,16] that the underlying model.generate returns.
    captured = {}
    orig_generate = model.model.generate

    def spy_generate(*a, **kw):
        codes_list, hidden_list = orig_generate(*a, **kw)
        captured["codes"] = [c.detach().cpu() for c in codes_list]
        return codes_list, hidden_list

    model.model.generate = spy_generate

    print("[ref_gen] generating (greedy) ...", flush=True)
    wavs, out_sr = model.generate_voice_clone(
        text=args.text,
        language=args.language,
        voice_clone_prompt=prompt,
        do_sample=False,
        subtalker_dosample=False,
        temperature=1.0,
        top_k=0,
        top_p=1.0,
        repetition_penalty=1.0,
        subtalker_temperature=1.0,
        subtalker_top_k=0,
        subtalker_top_p=1.0,
        max_new_tokens=args.max_new_tokens,
    )

    codes = captured["codes"][0].to(torch.int64).numpy()  # [T,16]
    np.save(os.path.join(args.outdir, "golden_codes.npy"), codes)
    with open(os.path.join(args.outdir, "golden_codes.txt"), "w") as f:
        f.write(f"# shape {codes.shape} (T, 16 codebooks)\n")
        for row in codes:
            f.write(" ".join(str(int(x)) for x in row) + "\n")

    out_wav = np.asarray(wavs[0], dtype=np.float32)
    sf.write(os.path.join(args.outdir, "golden_out.wav"), out_wav, out_sr)

    print(f"[ref_gen] DONE", flush=True)
    print(f"  talker_codes shape = {codes.shape}", flush=True)
    print(f"  codes[:3] =\n{codes[:3]}", flush=True)
    print(f"  wav samples = {out_wav.shape[0]} @ {out_sr} Hz "
          f"({out_wav.shape[0] / out_sr:.2f}s)", flush=True)


if __name__ == "__main__":
    main()
