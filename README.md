# Qwen3-TTS-ncnn

Port of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (12 Hz variant) to the [ncnn](https://github.com/Tencent/ncnn) inference framework — a pure-C++ text→speech pipeline with minimal third-party dependencies, cross-platform via CMake.

Entry for ncnn issue [#6791](https://github.com/Tencent/ncnn/issues/6791) (Tencent Rhino-Bird 2026).

## Overview

The full pipeline runs end to end in C++/ncnn and reproduces the PyTorch reference at every stage (see [Parity](#parity)):

```
text -> BPE tokenizer -> prefill assembly -> Talker (28-layer Qwen3 LM, autoregressive)
     -> Code Predictor (5-layer, nested 15-pass per frame)
     -> Speech Decoder (RVQ dequant + 8-layer transformer + causal ConvNet, 1920× upsample)
     -> 24 kHz mono WAV
```

- **Target: the 12 Hz checkpoints.** The 12 Hz path uses a convolutional RVQ decoder producing 24 kHz PCM directly — no diffusion, no BigVGAN.
- **Variant: CustomVoice.** A voice is a preset speaker (a codec token id), so there is no ECAPA-TDNN speaker encoder and no mel/FFT front-end.


## Status

| Stage | Component | State |
|-------|-----------|-------|
| P0 | PyTorch reference + deterministic golden dump | ✅ done |
| P2 | Talker (28-layer) → ncnn, kv-cache patch | ✅ cosine-sim 1.000000 |
| P3 | Code Predictor (5-layer) → ncnn | ✅ cosine-sim 1.000000 |
| P4 | Speech Decoder (transformer + conv stack + RVQ) → ncnn | ✅ cosine-sim 1.000000 |
| — | BPE tokenizer (vendored) | ✅ token-ids exact vs `Qwen2Tokenizer` |
| — | Prefill assembly (text → embeds) | ✅ max-abs 2.5e-5 vs PyTorch |
| — | C++ AR decode loop (Talker + Code Predictor + KV cache) | ✅ teacher-forced exact |
| — | **Self-contained text → WAV binary** | ✅ working end to end |
| — | Windows build | ⏳ pending (Linux ✅) |
| — | 1.7B scale + fp16/int8 | ⏳ optional |

## Repository layout

```
src/               C++ inference: pipeline, prefill assembly, RoPE, SnakeBeta, WAV I/O
src/tokenizer/     vendored byte-level BPE tokenizer
tests/             one numerical-parity harness per stage (ctest)
tools/convert/     pnnx conversion + post-processing scripts (Python)
docs/              technical write-up (EN/ZH) + reverse-engineered forward-path maps
configs/           model hyper-parameters
tests/golden/      deterministic reference codes / token ids for parity checks
```

Model weights and derived tables are **not** committed — they are regenerated from the official checkpoint by the scripts in `tools/convert/`.

## Conversion pipeline

Each neural stage is converted with `pnnx.export`. The transformer stacks need three non-obvious post-processing passes (pnnx does not emit any of them):

1. **mRoPE → plain RoPE** (`verify_mrope_collapse.py`) — the Talker's interleaved mRoPE provably collapses to a single cos/sin table in the text path (bit-exact), so it is monkeypatched to plain rotary embedding before tracing, keeping the exported graph clean.
2. **`dynamize_seqlen.py`** — the static pnnx export bakes the trace-time sequence length into the reshape ops; this rewrites the sequence dimension to `-1` so the graph accepts variable prefill length and single-token decode.
3. **`add_kvcache.py`** — enables the ncnn `SDPA` static-graph KV cache (`kv_cache=1`) and wires the per-layer `cache_k*/cache_v*` blobs. Ported from [futz12/ncnn_llm](https://github.com/futz12/ncnn_llm).

The vocoder's conv stack lowers to native ncnn ops; only the **SnakeBeta** activation ships as a small custom `ncnn::Layer`. RVQ dequant is folded into gather tables offline (`extract_rvq_tables.py`) so runtime dequant is pure gather-and-sum.

## Build

CMake links ncnn via `find_package(ncnn)` or an in-tree build via `-DNCNN_BUILD_DIR`. **C++17** is required (the tokenizer uses `std::optional`).

```bash
cmake -S . -B build -DNCNN_BUILD_DIR=/path/to/ncnn/build
cmake --build build
ctest --test-dir build           # per-stage parity harnesses
```

## Reproduce (from the official checkpoint)

```bash
# 1. environment (uv recommended; torch build must match your GPU arch)
uv venv .venv-tts --python 3.12
uv pip install --python .venv-tts qwen-tts pnnx ncnn
uv pip install --python .venv-tts torch --index-url https://download.pytorch.org/whl/cu128

# 2. download the checkpoint
hf download Qwen/Qwen3-TTS-12Hz-0.6B-CustomVoice \
    --local-dir models/Qwen3-TTS-12Hz-0.6B-CustomVoice

# 3. convert the neural stages
python tools/convert/convert_talker.py            --model models/Qwen3-TTS-12Hz-0.6B-CustomVoice --outdir nets
python tools/convert/convert_code_predictor.py    --model models/Qwen3-TTS-12Hz-0.6B-CustomVoice --outdir nets
python tools/convert/convert_decoder_transformer.py --model models/Qwen3-TTS-12Hz-0.6B-CustomVoice --outdir nets
python tools/convert/convert_decoder_convstack.py --model models/Qwen3-TTS-12Hz-0.6B-CustomVoice --outdir nets
python tools/convert/extract_rvq_tables.py        --model models/Qwen3-TTS-12Hz-0.6B-CustomVoice --outdir nets

# 4. synthesize (self-contained text → WAV)
./build/qwen3_tts nets assets out.wav "这是一个测试。" serena chinese
```

`assets/` holds `vocab.txt`, `merges.txt`, the projected text table and the embedding/head weight tables (regenerated by the convert scripts).

## Parity

Bit-exact code matching through greedy decoding is neither achievable nor the right metric — greedy `argmax` amplifies the sub-0.1 % numerical noise any fp32 conversion introduces. Parity is judged by **per-stage cosine similarity** on the pre-sampling activations and by **teacher forcing** (see the write-up):

| Stage | Metric | Result |
|-------|--------|--------|
| BPE tokenizer | token-id match | 22 / 22 |
| Prefill assembly | max-abs vs golden embeds | 2.5e-5 |
| Talker / Code Predictor / decoder transformer | cosine sim | 1.000000 |
| Full vocoder (golden codes) | cosine sim / max-abs | 1.000000 / 7.2e-5 |
| AR loop (teacher-forced) | cb0 match per frame | all frames |

## Acknowledgements

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — the model (Apache-2.0)
- [ncnn](https://github.com/Tencent/ncnn) — inference framework (BSD-3-Clause)
- [futz12/ncnn_llm](https://github.com/futz12/ncnn_llm) — LLM-on-ncnn reference and the KV-cache patch approach

## License

Apache-2.0 (see [LICENSE](LICENSE)), matching the upstream Qwen3-TTS model.
