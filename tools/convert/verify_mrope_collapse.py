# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# Verify the mRoPE-collapse hypothesis for the Qwen3-TTS Talker.
#
# Claim (from NOTES_talker_forward_map.md): in the TTS text path, get_rope_index
# makes all 3 mRoPE position rows equal, so apply_interleaved_rope collapses to
# row 0 -> a single cos/sin table suffices, and we can feed plain RoPE.
#
# This script proves it numerically: build the real mRoPE cos/sin the way the
# model does, run the REAL attention layer (full interleaved path), then run a
# mRoPE-FREE variant that applies plain rotate_half RoPE with the reduced table,
# and compare. If they match, the clean pnnx conversion path is unlocked.

import argparse
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--seq", type=int, default=6)
    args = ap.parse_args()

    from qwen_tts import Qwen3TTSModel
    import qwen_tts.core.models.modeling_qwen3_tts as M

    print("[mrope] loading (fp32, cpu) ...", flush=True)
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa"
    )
    talker = wrapper.model.talker
    cfg = talker.config
    layer = talker.model.layers[0].eval()
    rotary = talker.model.rotary_emb

    H = cfg.hidden_size
    head_dim = getattr(cfg, "head_dim", H // cfg.num_attention_heads)
    S = args.seq
    mrope_section = cfg.rope_scaling["mrope_section"]
    print(f"[mrope] head_dim={head_dim} mrope_section={mrope_section} "
          f"interleaved={cfg.rope_scaling.get('interleaved')}", flush=True)

    torch.manual_seed(0)
    hidden = torch.randn(1, S, H, dtype=torch.float32)
    mask = torch.triu(torch.full((S, S), float("-inf")), diagonal=1).view(1, 1, S, S)

    # Build mRoPE cos/sin exactly as the model: 3 equal position rows (text path).
    # rotary_emb.forward expects (x, position_ids) with position_ids shape (3,B,S).
    pos = torch.arange(S).view(1, 1, S).expand(3, 1, S).contiguous()
    with torch.no_grad():
        cos_full, sin_full = rotary(hidden, pos)  # each (3,B,S,head_dim)
    print(f"[mrope] cos_full shape={tuple(cos_full.shape)}", flush=True)

    # ---- REAL path: full interleaved mRoPE inside the layer ----
    with torch.no_grad():
        y_real = layer(hidden_states=hidden, attention_mask=mask,
                       position_embeddings=(cos_full, sin_full), use_cache=False)[0]

    # ---- REDUCED path: emulate what the C++/ncnn host will feed ----
    # apply_interleaved_rope collapses to row 0 when rows are equal; verify then
    # feed a mRoPE-free layer that uses plain rotate_half with the reduced table.
    def apply_interleaved_rope(x, mrope_section, modality_num):
        x_t = x[0].clone()
        for i, n in enumerate(mrope_section[1:], 1):
            beg, end = i, n * modality_num
            x_t[..., beg:end:modality_num] = x[i, ..., beg:end:modality_num]
        return x_t

    dim = cos_full.shape[-1]
    mn = len(mrope_section)
    cos_red = torch.cat([apply_interleaved_rope(cos_full[..., :dim // 2], mrope_section, mn)] * 2, dim=-1)
    sin_red = torch.cat([apply_interleaved_rope(sin_full[..., :dim // 2], mrope_section, mn)] * 2, dim=-1)
    # cos_red now (B,S,head_dim). Compare against just row-0 of the naive cat:
    cos_row0 = torch.cat([cos_full[0, ..., :dim // 2]] * 2, dim=-1)
    sin_row0 = torch.cat([sin_full[0, ..., :dim // 2]] * 2, dim=-1)
    collapse_ok = torch.allclose(cos_red, cos_row0, atol=1e-6) and torch.allclose(sin_red, sin_row0, atol=1e-6)
    print(f"[mrope] interleave collapses to row0: {collapse_ok} "
          f"(max diff cos={ (cos_red-cos_row0).abs().max().item():.2e})", flush=True)

    # Now run a mRoPE-free layer: monkeypatch apply_multimodal_rotary_pos_emb to a
    # plain rotate_half using the reduced (B,S,head_dim) table, unsqueezed to head dim.
    def plain_rope(q, k, cos, sin, mrope_section=None, mrope_interleaved=None, unsqueeze_dim=1):
        c = cos.unsqueeze(unsqueeze_dim)
        s = sin.unsqueeze(unsqueeze_dim)
        q2 = (q * c) + (M.rotate_half(q) * s)
        k2 = (k * c) + (M.rotate_half(k) * s)
        return q2, k2

    orig = M.apply_multimodal_rotary_pos_emb
    M.apply_multimodal_rotary_pos_emb = plain_rope
    try:
        with torch.no_grad():
            y_reduced = layer(hidden_states=hidden, attention_mask=mask,
                              position_embeddings=(cos_red, sin_red), use_cache=False)[0]
    finally:
        M.apply_multimodal_rotary_pos_emb = orig

    max_diff = (y_real - y_reduced).abs().max().item()
    rel = max_diff / (y_real.abs().max().item() + 1e-9)
    print(f"[mrope] REAL vs REDUCED layer output: max_abs_diff={max_diff:.3e} rel={rel:.3e}", flush=True)
    print(f"[mrope] VERDICT: {'PASS — reduced single-table RoPE is equivalent' if max_diff < 1e-4 else 'FAIL'}", flush=True)


if __name__ == "__main__":
    main()
