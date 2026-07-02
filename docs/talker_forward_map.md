I now have a complete and verified picture of the entire forward/generation path. Here is the structural map.

---

# Qwen3-TTS forward/generation map (`modeling_qwen3_tts.py`, 2299 lines)

Verified against the real checkpoint config at `/home/wwl/ncnn/models/Qwen3-TTS-12Hz-0.6B-Base/config.json`. Note the config.json values override the Python defaults: **`num_code_groups=16`**, **talker `num_hidden_layers=28`**, **code-predictor `num_hidden_layers=5`**, **`codec_eos_token_id=2150`** (the 2150 in your prompt is correct — the `4198`/`4196` in the Python file are stale defaults, not used).

## 1. nn.Module classes and containment

Top-level: **`Qwen3TTSForConditionalGeneration`** (L1813) holds two children:
- `self.talker` → `Qwen3TTSTalkerForConditionalGeneration` (L1820)
- `self.speaker_encoder` → `Qwen3TTSSpeakerEncoder` (L1823, ECAPA-TDNN x-vector extractor; not on the AR path — only used to produce a speaker embedding vector for voice cloning)
- `self.speech_tokenizer` (loaded externally, L1827/1843) — the vocoder/codec decoder, a separate module tree in `tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`.

**The Talker (main LM)** — `Qwen3TTSTalkerForConditionalGeneration` (L1564), base_model_prefix `"talker"`:
- `self.model` → `Qwen3TTSTalkerModel` (L1573 / def L1427) — the 28-layer Qwen3 decoder.
  - `self.layers` = 28 × `Qwen3TTSTalkerDecoderLayer` (L1435, def L1348)
    - each: `self_attn` = `Qwen3TTSTalkerAttention` (L1352 / def L727), `mlp` = `Qwen3TTSTalkerTextMLP` (L1354 / def L842), `input_layernorm` + `post_attention_layernorm` = `Qwen3TTSRMSNorm` (L596).
  - `self.norm` = `Qwen3TTSRMSNorm` (L1438)
  - `self.rotary_emb` = `Qwen3TTSTalkerRotaryEmbedding` (L1439 / def L526) — the mRoPE generator (3-way expand).
  - **`self.codec_embedding`** = `nn.Embedding(vocab_size=3072, hidden_size=1024)` (L1441) — the codebook-0 / semantic-token embedding table + all the special codec tokens (bos/pad/eos/think/language/speaker ids all index into this same table).
  - **`self.text_embedding`** = `nn.Embedding(text_vocab_size=151936, text_hidden_size=2048)` (L1442) — input text token table.
- `self.text_projection` → `Qwen3TTSTalkerResizeMLP` (L1575 / def L808) — projects 2048-d text embeds to 1024-d hidden (fc1→act→fc2, bias=True).
- **`self.codec_head`** = `nn.Linear(1024 → 3072, bias=False)` (L1579) — the Talker's LM head over the codec vocab (this produces the codebook-0 logits).
- **`self.code_predictor`** → `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration` (L1580) — the nested "sub-talker"/MTP.

**The Code Predictor (sub-talker / MTP)** — `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration` (L1156), base_model_prefix `"talker.code_predictor"`:
- `self.model` → `Qwen3TTSTalkerCodePredictorModel` (L1165 / def L1015) — a small 5-layer decoder.
  - `self.layers` = 5 × **`Qwen3TTSDecoderLayer`** (L1023, def L961) — note: a *different* layer class than the Talker's. Its attn is `Qwen3TTSAttention` (L966 / def L885) which uses plain 1-D `apply_rotary_pos_emb` (L858), **not** mRoPE.
  - `self.norm` = `Qwen3TTSRMSNorm` (L1026)
  - `self.rotary_emb` = `Qwen3TTSRotaryEmbedding` (L1027 / def L561) — standard 1-D RoPE.
  - **`self.codec_embedding`** = `nn.ModuleList([nn.Embedding(2048, 1024) for _ in range(num_code_groups-1)])` = **15 embedding tables** (L1030), one per codebook 1..15. `get_input_embeddings()` returns this list (L1037).
- **`self.lm_head`** = `nn.ModuleList([nn.Linear(1024 → 2048, bias=False) for _ in range(15)])` = **15 output heads** (L1167), one per codebook 1..15.
- `self.small_to_mtp_projection` (L1171) — `Identity()` here because talker hidden_size == code_predictor hidden_size (both 1024).

Other module classes: RMSNorm (L596), two rotary embeddings (L526, L561), speaker-encoder sub-blocks Res2NetBlock/SqueezeExcitationBlock/AttentiveStatisticsPooling/TimeDelayNetBlock/SqueezeExcitationRes2NetBlock (L95–309).

## 2. The Talker decode step (`Qwen3TTSTalkerForConditionalGeneration.forward`, L1636)

The Talker's AR loop is driven by HF `GenerationMixin.generate`, invoked at L2272 with `inputs_embeds=`. **The Talker consumes embeddings, not token ids, on the input side** — but each decode step also receives the previously sampled codebook-0 `input_ids` (via HF's standard loop) which it re-embeds internally.

Two branches inside `forward`:
- **Prefill** (`inputs_embeds.shape[1] > 1`, L1665): `generation_step=-1`, `codec_ids=None`, uses the prompt embeds directly.
- **Generate** (single step, L1669+): this is where the frame is assembled — see section 3.

Position ids / mRoPE (L1693–1711):
- First step (`cache_position[0]==0`): `position_ids, rope_deltas = self.get_rope_index(attention_mask)` (L1746). `get_rope_index` builds `position_ids = attention_mask.float().cumsum(-1) - 1`, masks pads to 1, then `unsqueeze(0).expand(3,-1,-1)` → shape **`(3, B, S)`** (L1794–1796). So the three mRoPE sections all get the *same* cumulative text position here (this is effectively text-style positioning).
- Subsequent steps (L1706–1711): `position_ids = arange(seq_length) + (cache_position[0] + rope_deltas)`, expanded to `(3, B, S)`.
- Inside `Qwen3TTSTalkerModel.forward`, `position_ids` of ndim 3 is kept as-is; `text_position_ids = position_ids[0]` (L1508) is used for the causal mask, while the full 3-row tensor goes to `rotary_emb` (L1523). `position_id_per_seconds=13` from config is **not referenced anywhere in this modeling file** (grep confirms zero hits) — it is metadata only, not applied in this code path.

mRoPE application — `Qwen3TTSTalkerRotaryEmbedding.forward` (L546): expands `inv_freq` to `(3, positions, dim/2, 1)` and matmuls with `position_ids (3,B,1,pos)` → per-section cos/sin of shape `(3, B, pos, dim)`. Then `Qwen3TTSTalkerAttention.forward` (L778) calls `apply_multimodal_rotary_pos_emb(..., mrope_section=[24,20,20], interleaved=True)` (L660). Since `interleaved=True`, the `apply_interleaved_rope` inner function (L694–703) is used — see section 6.

KV cache: a HF `DynamicCache` (L1487), passed into every layer; `past_key_values.update(key, value, layer_idx, {sin,cos,cache_position})` at L785.

Attention implementation: dispatched at L787–789 — `eager_attention_forward` (L634) unless `config._attn_implementation != "eager"`, in which case `ALL_ATTENTION_FUNCTIONS[self.config._attn_implementation]`. The checkpoint config sets no override, and HF default is **sdpa**. The eager path (L634–657) is standard: `repeat_kv` for GQA (16 q-heads / 8 kv-heads → n_rep=2), `softmax(QKᵀ·scaling + mask)·V`. Q/K get per-head RMSNorm (`q_norm`/`k_norm`, L773–774) before RoPE — Qwen3 style.

Output: `hidden_states = model(...)`, then `logits = self.codec_head(hidden_states)` (L1727) over codec vocab **3072**. Sampling of codebook-0 is done by the HF generate loop (top_k=50, top_p, temperature=0.9, repetition_penalty=1.05, plus `suppress_tokens` on the top-1024 codec vocab range except eos, L2059).

## 3. THE CRITICAL PART — Talker ↔ Code Predictor aggregation (L1669–1692)

This is the exact per-frame flow, quoted. On generation step, `input_ids` = the codebook-0 token the Talker just sampled at the previous step:

```python
1670  last_id_hidden = self.get_input_embeddings()(input_ids)          # embed cb0 token via talker.codec_embedding → [B,1,1024]
1671  predictor_result = self.code_predictor.generate(
1672      inputs_embeds=torch.cat((past_hidden, last_id_hidden), dim=1), # [B,2,1024]: prev-frame talker hidden + cb0 embed
1673      max_new_tokens=self.config.num_code_groups - 1,               # = 15
1674      do_sample=subtalker_dosample, top_p=..., top_k=..., temperature=...,
1678      output_hidden_states=True, return_dict_in_generate=True,
1679  )
1681  codec_ids = torch.cat((input_ids, predictor_result.sequences), dim=-1)   # [B,16]: cb0 ++ cb1..15
1682  codec_hiddens = torch.cat(
1683      [last_id_hidden]
1684      + [self.code_predictor.get_input_embeddings()[i](predictor_result.sequences[..., i:i+1]) for i in range(self.config.num_code_groups - 1)],
1686      dim=1,                                                         # [B,16,1024]
1687  )
1688  inputs_embeds = codec_hiddens.sum(1, keepdim=True)                 # [B,1,1024]  ← SUM over all 16 codebook embeds
1690  inputs_embeds = inputs_embeds + trailing_text_hidden[:, generation_step] (or + tts_pad_embed)  # add text-stream hidden
```

Precise trace:

**(a) What crosses Talker→CodePredictor:** the concatenation `[past_hidden, last_id_hidden]` (L1672), a length-2 sequence of 1024-d vectors:
- `past_hidden` = the Talker's *last hidden state from the previous frame* (`hidden_states[:, -1:, :]`, set at L1740 and threaded via `_update_model_kwargs_for_generation` L1806). It is the **hidden state, not logits**.
- `last_id_hidden` = `talker.codec_embedding(cb0_token)` (L1670) — the embedding of the codebook-0 token the Talker sampled.

**(b) The 15 sequential passes** happen *inside* `self.code_predictor.generate` (HF loop, max_new_tokens=15). Each step lands in `Qwen3TTSTalkerCodePredictorModelForConditionalGeneration.forward` (L1250):
- **Prefill of the sub-talker** (L1277): the 2-token `inputs_embeds` sets `generation_steps = inputs_embeds.shape[1] - 2 = 0`. `small_to_mtp_projection` = Identity (L1282). It runs the 5-layer model, then `logits = self.lm_head[0](hidden_states)` (L1299) → predicts **codebook 1** from the last position. Returns `generation_steps = 1` (L1311).
- **Each subsequent step k (k=1..14)** (L1281): `inputs_embeds = self.model.get_input_embeddings()[generation_steps-1](input_ids)` — i.e. the just-sampled code for codebook k is embedded with **`code_predictor.codec_embedding[k-1]`**, appended to the sub-talker's own KV cache, and `logits = self.lm_head[generation_steps](hidden_states)` (L1299) predicts codebook k+1. `generation_steps` increments each call.
- So head/embedding indexing is staggered: `lm_head[i]` and `codec_embedding[i]` correspond to codebook `i+1`. The sub-talker uses its own `DynamicCache`, standard 1-D RoPE, position_ids = cache_position (L1091–1092).

**(c) Feedback of each predicted code:** handled by HF generate appending `sequences`; each new code is re-embedded at the next step's L1281 (through `codec_embedding[generation_steps-1]`). Result `predictor_result.sequences` = `[B,15]` codes for codebooks 1..15.

**(d) Assembly of the 16 codes for the frame:** L1681 `codec_ids = cat(cb0_input_ids, sequences)` → **`[B,16]`**, order `[cb0, cb1, ..., cb15]`.

**(e) Feeding the frame back into the Talker:** L1682–1687 re-embeds all 16 codes — cb0 with `talker.codec_embedding` (reusing `last_id_hidden`), cb1..15 with `code_predictor.codec_embedding[i]` — stacks to `[B,16,1024]` and **sums across the 16 codebooks** (`.sum(1, keepdim=True)`, L1687) to make one 1024-d vector. Then the **text stream** is added: `+ trailing_text_hidden[:, generation_step]` while text remains (L1689–1690) else `+ tts_pad_embed` (L1692). That sum is the Talker's `inputs_embeds` for the next decode step. This additive fusion (codec-sum + text-hidden) is the same scheme used in prefill (see L2182–2202 where text_projection(text) + codec_embedding are added) and in `generate_icl_prompt` (L1989 sums the 16 ref-code embeds).

Key risk notes for your port: the cross-model handoff is the **hidden state** `past_hidden` (previous frame) concatenated with the **cb0 embedding** (current frame) — a 2-token prefill, not a sum. The feedback INTO the talker is a **sum of 16 code-embeddings + one text hidden**. Two different aggregation rules; don't conflate them.

## 4. Per-frame codes → final tensor (L2280–2292)

After `self.talker.generate` returns, in `Qwen3TTSForConditionalGeneration.generate`:
```python
2280  talker_codes = torch.stack([hid[-1] for hid in talker_result.hidden_states if hid[-1] is not None], dim=1)
```
Each step stored `hidden_states=(outputs.hidden_states, codec_ids)` (L1738), so `hid[-1]` = that step's `codec_ids` `[B,16]`. Stacking over `dim=1` gives **`talker_codes` shape `[B, T, 16]`** — layout **[T, 16]** per sample (time-major, 16 codebooks last).

- `talker_hidden_states` `[B,T,1024]` also collected (L2281) — returned but unused by the 12Hz vocoder path.
- Truncation at EOS: `first_codebook = talker_codes[:, :, 0]` (L2283), `is_stop = (first_codebook == codec_eos_token_id)` (L2284), `effective_lengths` via argmax of the stop mask (L2285–2287). Per-sample slice `talker_codes[i, :length]` → list of `[T_i, 16]` tensors (L2289).

Handoff to vocoder: `qwen3_tts_model.py` L620 calls `speech_tokenizer.decode([{"audio_codes": c}])` with each `c` = `[T,16]`. The tokenizer decode (`modeling_qwen3_tts_tokenizer_v2.py` L993) expects `audio_codes` shape `(B, codes_length, num_quantizers)` = `(B,T,16)`, then does `audio_codes.transpose(1,2)` → `(B,16,T)` before `chunked_decode` (L1015). The decoder core `forward` asserts `codes.shape[1] == num_quantizers (16)` (L870). So: **the AR loop produces [T,16]; the vocoder internally transposes to [16,T].**

## 5. Stopping condition

The outer Talker loop is HF `generate` with (L2044–2058): `max_new_tokens=4096` (default; overridable), `min_new_tokens=2`, and `eos_token_id = codec_eos_token_id = 2150`. HF stops when the sampled **codebook-0** token equals 2150. Post-hoc, the code also truncates each sequence at the first cb0==2150 frame (L2283–2287) in case of batch/left-padding. There is no separate stop on codebooks 1..15.

## 6. Custom ops / tracing hazards

- **Interleaved mRoPE** (`apply_multimodal_rotary_pos_emb`, L660; inner `apply_interleaved_rope`, L694–703). With `interleaved=True` it does strided in-place channel scatter: `x_t[..., beg:end:modality_num] = x[beg, ..., beg:end:modality_num]` (L702), looping over sections `[24,20,20]` with `modality_num=3`. This strided-assignment-with-clone is data-dependent indexing that pnnx/ONNX tracing will not reproduce faithfully — you must hand-implement the section interleave in C++. Note it reads `x[beg_idx, ...]` (indexing the 3-section axis) — the three position rows differ only when position_ids rows differ, which in this path they generally don't (get_rope_index makes all 3 rows equal), so the interleave largely collapses to the row-0 values — verify numerically but you can likely precompute a single cos/sin per position.
- `rotate_half` (L615) — standard, fine.
- `repeat_kv` (L622) via `expand`+`reshape` for GQA (n_rep=2) — implement as head broadcast.
- **The `.sum(1)` codebook aggregation** (L1687, L1989, L2189-implicit) — trivial in C++ but semantically critical.
- Dynamic two-nested `generate` loops with per-step `generation_steps`/`generation_step` counters selecting **which embedding table and which lm_head** (L1281, L1299) — this indexed module selection cannot be traced; you must unroll 15 heads/embeddings explicitly.
- Two independent KV caches (talker 28-layer mRoPE cache; code-predictor 5-layer 1-D-RoPE cache that is **rebuilt fresh every frame** — the sub-talker cache does not persist across frames, only across the 15 inner steps).
- Left-padding / `pad_sequence` with flip (L2242–2249) and `get_rope_index` cumsum positioning (L1794) — batch-only; for a single-utterance C++ port you can use a simple `[0,1,2,...]` position sequence.
- RMSNorm forced to float32 (L607), rotary forced to float32 (L553) — match precision.

### Convenient ncnn submodule split
1. **Talker embed/prefill assembler** (text_embedding → text_projection, codec_embedding, additive fusion) — mostly host code + two embedding gathers + one small MLP.
2. **Talker decoder block ×28** (RMSNorm, q/k/v proj, q/k-norm, interleaved mRoPE, GQA sdpa, MLP) → one ncnn graph run per frame with external KV cache.
3. **Talker codec_head** (1024→3072 linear) + sampling on host.
4. **Code-predictor decoder block ×5** (1-D RoPE) → run 15× per frame with its own KV cache; 15 embedding tables + 15 linear heads as weights, selected on host.
5. **Aggregation glue** (sum of 16 embeds + text hidden) on host.
6. Speech tokenizer / vocoder is a separate model tree (`tokenizer_12hz/...v2.py`), consumes `[T,16]`.

Relevant files:
- `/home/wwl/ncnn/.venv-tts/lib/python3.12/site-packages/qwen_tts/core/models/modeling_qwen3_tts.py`
- `/home/wwl/ncnn/.venv-tts/lib/python3.12/site-packages/qwen_tts/core/models/configuration_qwen3_tts.py`
- `/home/wwl/ncnn/models/Qwen3-TTS-12Hz-0.6B-Base/config.json` (authoritative hyperparameters)
- `/home/wwl/ncnn/.venv-tts/lib/python3.12/site-packages/qwen_tts/inference/qwen3_tts_model.py` (L603–620: generate→decode wiring)
- `/home/wwl/ncnn/.venv-tts/lib/python3.12/site-packages/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py` (L869–884, L993–1015: code layout `[B,T,16]`→transpose→`[B,16,T]`)