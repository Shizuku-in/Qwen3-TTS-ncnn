// SPDX-License-Identifier: Apache-2.0
// Host-side RoPE cos/sin cache builders.
//
// Qwen3-TTS uses three distinct RoPE thetas across its components:
//   - Talker backbone:      theta = 1e6,  head_dim 128 (mRoPE, but proven to
//                           collapse to a single plain-RoPE table in the TTS path)
//   - Code Predictor:       theta = 1e6,  head_dim 128 (standard 1-D RoPE)
//   - Speech decoder tf:    theta = 1e4,  head_dim 64  (standard 1-D RoPE)
//
// Each builds a (seqlen, head_dim) cos and sin table the ncnn RotaryEmbed layer
// consumes. Layout matches what the converted nets expect: ncnn Mat (w=head_dim,
// h=seqlen), i.e. one row per position.
#pragma once

#include "mat.h"
#include <math.h>

namespace qwen3_tts {

// Build cos/sin caches for positions [pos_offset, pos_offset+seqlen).
// half = head_dim/2 frequencies, duplicated into the full head_dim
// (standard HF layout: emb = cat(freqs, freqs)).
static inline void build_rope_cache(int seqlen, int head_dim, float theta,
                                    int pos_offset, ncnn::Mat& cos_cache,
                                    ncnn::Mat& sin_cache)
{
    const int half = head_dim / 2;
    cos_cache.create(head_dim, seqlen);
    sin_cache.create(head_dim, seqlen);

    for (int s = 0; s < seqlen; s++)
    {
        const float pos = (float)(pos_offset + s);
        float* cptr = cos_cache.row(s);
        float* sptr = sin_cache.row(s);
        for (int i = 0; i < half; i++)
        {
            const float inv_freq = powf(theta, -(float)(2 * i) / (float)head_dim);
            const float angle = pos * inv_freq;
            const float c = cosf(angle);
            const float sv = sinf(angle);
            // duplicated halves: index i and i+half share the same angle
            cptr[i] = c;
            cptr[i + half] = c;
            sptr[i] = sv;
            sptr[i + half] = sv;
        }
    }
}

} // namespace qwen3_tts
