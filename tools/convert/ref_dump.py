#!/usr/bin/env python3
# Copyright 2026
# SPDX-License-Identifier: Apache-2.0
#
# P0 reference harness for the Qwen3-TTS -> ncnn port (ncnn issue #6791).
#
# Purpose:
#   Run the official PyTorch Qwen3-TTS (qwen-tts package) in a DETERMINISTIC mode
#   and dump intermediate tensors at every stage boundary, so the ncnn C++ port
#   can be validated numerically stage-by-stage ("parity gate").
#
# Why deterministic:
#   The model samples (temperature/top-k/top-p). Bit-exact end-to-end audio match
#   is impossible through stochastic sampling. So for parity we force greedy
#   (do_sample=False) and compare PRE-SAMPLING logits / distributions and the
#   deterministic vocoder waveform, not sampled token ids.
#
# This script only READS the model; it writes dumps under tools/qwen3_tts/dumps/.
#
# Usage:
#   .venv-tts/bin/python tools/qwen3_tts/scripts/ref_dump.py \
#       --model models/Qwen3-TTS-12Hz-0.6B-Base \
#       --ref-audio <ref.wav> --ref-text "<transcript>" \
#       --text "你好，世界。" --language Chinese \
#       --out tools/qwen3_tts/dumps/run0
#
# Note: 0.6B-Base is a voice-clone model, so --ref-audio + --ref-text are required.
#       For a CustomVoice checkpoint use --speaker instead (see --help).

import argparse
import json
import os
import sys

import numpy as np
import torch


def log(*a):
    print("[ref_dump]", *a, file=sys.stderr, flush=True)


# --------------------------------------------------------------------------
# Dump helpers: save tensors in a format the C++/ncnn side can read back.
# We save both .npy (for python-side comparison) and a raw little-endian
# float32 .bin + a .json sidecar with shape/dtype (for C++ side).
# --------------------------------------------------------------------------
class Dumper:
    def __init__(self, outdir):
        self.outdir = outdir
        os.makedirs(outdir, exist_ok=True)
        self.manifest = {}

    def save(self, name, tensor):
        if isinstance(tensor, torch.Tensor):
            arr = tensor.detach().to(torch.float32).cpu().numpy()
        else:
            arr = np.asarray(tensor)
        # canonical npy
        np.save(os.path.join(self.outdir, name + ".npy"), arr)
        # raw f32 + sidecar for the C++ side
        arr_f32 = np.ascontiguousarray(arr, dtype="<f4")
        arr_f32.tofile(os.path.join(self.outdir, name + ".bin"))
        self.manifest[name] = {
            "shape": list(arr.shape),
            "dtype": "float32",
            "npy": name + ".npy",
            "bin": name + ".bin",
        }
        log(f"dumped {name} shape={list(arr.shape)}")

    def save_ids(self, name, ids):
        arr = np.asarray(ids, dtype="<i4")
        np.save(os.path.join(self.outdir, name + ".npy"), arr)
        arr.tofile(os.path.join(self.outdir, name + ".bin"))
        self.manifest[name] = {
            "shape": list(arr.shape),
            "dtype": "int32",
            "npy": name + ".npy",
            "bin": name + ".bin",
        }
        log(f"dumped {name} (int32) shape={list(arr.shape)}")

    def flush(self, extra=None):
        m = dict(self.manifest)
        if extra:
            m["_meta"] = extra
        with open(os.path.join(self.outdir, "manifest.json"), "w") as f:
            json.dump(m, f, indent=2, ensure_ascii=False)
        log(f"wrote manifest.json with {len(self.manifest)} tensors")


def build_argparser():
    p = argparse.ArgumentParser(description="Qwen3-TTS PyTorch reference dumper")
    p.add_argument("--model", required=True, help="path to model dir")
    p.add_argument("--text", required=True, help="text to synthesize")
    p.add_argument("--language", default="Chinese")
    p.add_argument("--ref-audio", default=None, help="reference wav (Base/voice-clone)")
    p.add_argument("--ref-text", default=None, help="reference transcript (Base ICL)")
    p.add_argument("--speaker", default=None, help="speaker name (CustomVoice models)")
    p.add_argument("--instruct", default=None, help="style instruction (VoiceDesign)")
    p.add_argument("--out", required=True, help="output dump directory")
    p.add_argument("--max-new-tokens", type=int, default=256,
                   help="cap frames for a quick deterministic run")
    p.add_argument("--device", default="cuda:0")
    p.add_argument("--seed", type=int, default=0)
    return p


def main():
    args = build_argparser().parse_args()
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)

    dumper = Dumper(args.out)

    log(f"loading model from {args.model} ...")
    import torch as _torch
    from qwen_tts import Qwen3TTSModel

    wrapper = Qwen3TTSModel.from_pretrained(
        args.model,
        device_map=args.device,
        dtype=_torch.bfloat16,
        attn_implementation="sdpa",  # sdpa (not flash) — matches pnnx conversion path
    )
    model = wrapper.model
    model.eval()
    log(f"model loaded: tts_model_type={model.tts_model_type} size={model.tts_model_size}")

    # Stage A dump: tokenized text ids (host-side frontend parity)
    input_texts = [wrapper._build_assistant_text(args.text)]
    input_ids_list = wrapper._tokenize_texts(input_texts)
    dumper.save_ids("A_input_ids", input_ids_list[0].reshape(-1).cpu().numpy())

    dumper.flush(extra={
        "model": os.path.abspath(args.model),
        "tts_model_type": model.tts_model_type,
        "tts_model_size": str(model.tts_model_size),
        "text": args.text,
        "language": args.language,
        "note": "Stage A only in this pass; AR-loop + vocoder dumps added after "
                "the Talker<->CodePredictor aggregation is locked down.",
    })
    log("done (stage A). Deeper stage hooks are added in the next iteration.")


if __name__ == "__main__":
    main()
