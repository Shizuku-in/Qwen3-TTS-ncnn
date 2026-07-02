# Qwen3-TTS 12Hz Speech Decoder (vocoder) — decode path map

Source: `.venv-tts/lib/python3.12/site-packages/qwen_tts/core/tokenizer_12hz/modeling_qwen3_tts_tokenizer_v2.py`
Authoritative dims: `models/Qwen3-TTS-12Hz-0.6B-Base/speech_tokenizer/config.json` (config.json overrides Python defaults — trust checkpoint tensor shapes).

## Entry & call chain
`Qwen3TTSTokenizerV2Model.decode()` (L993): input `audio_codes (B,T,16)`.
- L1012: `audio_lengths = (audio_codes[...,0] > -1).sum(1) * 1920` (output sample count from valid frames).
- L1014: `clamp(min=0)` — padding sentinel `-1` → 0.
- L1015: `decoder.chunked_decode(audio_codes.transpose(1,2))` → transpose to `(B,16,T)`; `.squeeze(1)` → `(B,samples)`.
- L1017: trim each waveform to audio_lengths.

Chain: decode → chunked_decode(886) → forward(869) → RVQ decode(815) → pre_conv(512→1024) → pre_transformer(8 layers)(501) → upsample ModuleList(4x) → decoder ModuleList(480x) → clamp(-1,1).

`chunked_decode` (886): pure Python windowing, chunk_size=300, left_context_size=25 frames. Each chunk = independent full forward; discards `context*1920` leading samples to hide zero-pad seam. **No state between chunks.** For port: can run whole sequence in ONE pass (simplest, exact) OR replicate chunk+overlap-trim for bounded memory. No recurrence.

## RVQ dequant (the custom part)
Split: semantic = first `n_q_semantic=1` codebook, acoustic = remaining 15. Separate `ResidualVectorQuantizer` modules; their `(B,512,T)` outputs **summed** (L818-820).
- Internal codebook vectors are **256-dim** (dimension = codebook_dim//2 = 256), lifted 256→512 by a per-group `output_proj` = **1x1 Conv1d bias=False** applied once after summation.
- Per-codebook contributions **summed** across K (L721-726).
- `EuclideanCodebook.decode` (L676): table computed at RUNTIME as `embedding_sum / clamp(cluster_usage, eps)` then `F.embedding`. **PRECOMPUTE offline** and bake into a gather/Embedding table for ncnn.
- **Codebook sizes: semantic 4096 (`rvq_first...embedding_sum`=[4096,256]), acoustic 2048 ([2048,256]).** Config passes single `bins` to both but real checkpoint shapes differ — trust tensor shapes. Semantic idx 0..4095, acoustic 0..2047.
- After quantizer: `pre_conv` = `CausalConvNet(512→1024, k=3)` lift to latent; transpose to `(B,T,1024)`.

## Transformer (pre_transformer, L501)
Input `(B,T,1024)`. `input_proj` Linear(1024→512). 8× layer: RMSNorm→self_attn→residual+layer_scale; RMSNorm→SwiGLU mlp→residual+layer_scale. final RMSNorm. `output_proj` Linear(512→1024)→ permute `(B,1024,T)`.
- Attention: head_dim=64 (from config.json; Python default 32 — VERIFY checkpoint), q/k/v = Linear(512→16*64=1024), o_proj Linear(1024→512). num_kv_heads=16 → **no GQA**. q_norm/k_norm = Identity (**no QK-norm** despite Qwen3 naming). **RoPE theta=10000**. All 8 layers **sliding_attention, window=72** → static causal sliding-window mask. MLP SwiGLU intermediate=1024 (config.json; Python default 3072).

## Upsample ModuleList (L845, ratios [2,2], total 4x)
Per factor: `CausalTransConvNet(1024→1024, k=factor, stride=factor)` → `ConvNeXtBlock(1024)`: depthwise CausalConvNet(k=7,groups=1024)→permute→LayerNorm(eps1e-6)→Linear(1024→4096)→GELU→Linear(4096→1024)→gamma*→permute→residual. Output `(B,1024,4T)`.

## Decoder ModuleList (L857, rates [8,5,4,3], total 480x)
- decoder[0]: CausalConvNet(1024→1536, k=7) → `(B,1536,4T)`.
- 4× DecoderBlock i (in=1536//2^i, out=1536//2^(i+1), rate=rates[i]): SnakeBeta → CausalTransConvNet(k=2*rate, stride=rate) → 3× ResidualUnit(dil=1,3,9).
  - i=0: 1536→768, k=16 s=8 → 32T; i=1: 768→384, k=10 s=5 → 160T; i=2: 384→192, k=8 s=4 → 640T; i=3: 192→96, k=6 s=3 → 1920T.
  - ResidualUnit (L619): SnakeBeta→CausalConvNet(k=7,dilation=d)→SnakeBeta→CausalConvNet(k=1)→+residual.
- SnakeBeta(96) → CausalConvNet(96→1, k=7) → `(B,1,1920T)` → clamp(-1,1).

Total upsample = 4 × 480 = 1920. ✓

## Custom ncnn layers needed
1. **CausalConvNet** (L159): dynamic causal left-pad = `kernel_size - stride`, plus runtime `extra_padding` via ceil. Implement as explicit asymmetric left-pad + plain Conv1d. Used everywhere.
2. **CausalTransConvNet** (L195): ConvTranspose1d + dynamic right-trim `[..., :len - right_pad]`, right_pad = k - stride. Implement transposed conv + fixed tail crop.
3. **SnakeBeta** (L578): `x + (1/(exp(beta)+1e-9)) * sin(x*exp(alpha))^2`, per-channel learnable alpha/beta. No ncnn equiv → custom.
4. **EuclideanCodebook**: precompute table offline, port as gather.
5. RMSNorm (L373) ×17 in transformer; LayerScale (L394) per-channel residual mul; ConvNeXtBlock (L211).
6. RoPE cos/sin + static sliding-window causal mask (window 72) precomputed on host.

**No weight_norm anywhere** — plain Conv1d/ConvTranspose1d. One fewer hazard than typical GAN vocoders.

## Output
24000 Hz, mono, float [-1,1] (clamp L884). No int16 conversion built in (multiply 32767 yourself). Length = num_frames * 1920.
