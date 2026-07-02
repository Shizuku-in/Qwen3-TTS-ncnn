# Qwen3-TTS-ncnn

Port of [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) (12 Hz variant) to the
[ncnn](https://github.com/Tencent/ncnn) inference framework — C++ inference with
minimal third-party dependencies, cross-platform via CMake.

This is a work-in-progress entry for ncnn issue
[#6791](https://github.com/Tencent/ncnn/issues/6791) (Tencent Rhino-Bird 2026).

## Goal

Run Qwen3-TTS end to end on ncnn and match the PyTorch reference audio given the
same input:

```
text ─► Qwen BPE tokenizer ─► Talker (28-layer Qwen3 LM, AR) ─► Code Predictor (5-layer MTP, nested AR)
     ─► Speech Decoder (RVQ + causal ConvNet, 1920× upsample) ─► 24 kHz WAV
```

Target: the **12 Hz** checkpoints (`Qwen3-TTS-12Hz-0.6B/1.7B`), which use a
convolutional RVQ decoder — no diffusion / BigVGAN.

## Status

| Stage | Component | State |
|-------|-----------|-------|
| P0 | PyTorch reference + deterministic golden dump | ✅ done |
| P2 | Talker (28-layer) → ncnn, kv-cache patch, parity | ✅ converted, cosine-sim 1.000 vs PyTorch |
| P3 | Code Predictor (5-layer) → ncnn, parity | ✅ converted, cosine-sim 1.000 vs PyTorch |
| P4 | Speech Decoder (RVQ + custom conv layers) | ⏳ next |
| P1/P5 | C++ AR decode loop + end-to-end + WAV | ⏳ pending |
| — | Windows + Linux CMake build | ⏳ pending |

## Repository layout

```
tools/convert/     pnnx conversion + post-processing scripts (Python)
docs/              reverse-engineered forward-path maps (Talker, Speech Decoder)
configs/           model hyper-parameters (from the 0.6B-Base checkpoint)
tests/golden/      deterministic reference codes for numerical parity checks
```

Model weights are **not** committed — they are regenerated from the official
checkpoints via the conversion scripts (see below).

## Conversion pipeline

The transformer backbones convert through pnnx with two required post-processing
passes (pnnx alone does not emit either):

1. **`dynamize_seqlen.py`** — the clean static pnnx export bakes the trace-time
   sequence length into the reshape ops. This rewrites the sequence dimension to
   `-1` so the graph accepts variable prefill length and single-token decode,
   matching the layout ncnn's shipped Qwen3 decoder uses.
2. **`add_kvcache.py`** — enables the ncnn `SDPA` static-graph KV cache
   (`kv_cache=1`) and wires the per-layer `cache_k*/cache_v*` in/out blobs. Ported
   from [futz12/ncnn_llm](https://github.com/futz12/ncnn_llm).

A key finding (`verify_mrope_collapse.py`): the Talker's interleaved mRoPE
provably collapses to a single cos/sin table in the text path, so it is replaced
by plain rotary embedding (bit-exact), keeping the exported graph clean.

### Reproduce

```bash
# 1. environment (uv recommended; torch build must match your GPU arch)
uv venv .venv-tts --python 3.12
uv pip install --python .venv-tts qwen-tts pnnx ncnn
#   torch matching your CUDA/arch, e.g. Blackwell sm_120 needs cu128+
uv pip install --python .venv-tts torch --index-url https://download.pytorch.org/whl/cu128

# 2. download an official checkpoint
hf download Qwen/Qwen3-TTS-12Hz-0.6B-Base --local-dir models/Qwen3-TTS-12Hz-0.6B-Base

# 3. convert the transformer backbones
python tools/convert/convert_talker.py         --model models/Qwen3-TTS-12Hz-0.6B-Base
python tools/convert/convert_code_predictor.py --model models/Qwen3-TTS-12Hz-0.6B-Base
```

## Acknowledgements

- [Qwen3-TTS](https://github.com/QwenLM/Qwen3-TTS) — the model (Apache-2.0)
- [ncnn](https://github.com/Tencent/ncnn) — inference framework (BSD-3-Clause)
- [futz12/ncnn_llm](https://github.com/futz12/ncnn_llm) — LLM-on-ncnn reference

## License

Apache-2.0 (see [LICENSE](LICENSE)), matching the upstream Qwen3-TTS model.
