// SPDX-License-Identifier: Apache-2.0
// CustomVoice Talker prefill assembly: token ids -> inputs_embeds + trailing.
//
// Ports the validated Python reproduction (repro_prefill.py, max_abs 2.4e-7 vs
// the dumped golden). The whole assembly is a deterministic gather + additive
// composition over two precomputed tables:
//   - proj_table[t]  = text_projection(text_embedding[t])   (VOCAB, 1024)  raw .f32
//   - codec_emb[c]    = talker.codec_embedding[c]             (3072, 1024)  raw .f32
// so there is NO matmul here (text_projection is baked into proj_table offline).
//
// Non-streaming CustomVoice layout (matches modeling_qwen3_tts.generate, the
// tts_model_type==custom_voice branch, chinese/language_id present):
//
//   role      = proj_table[ ids[0:3] ]                          # 3  <|im_start|>assistant\n
//   codec_pfx = codec_emb[ think, think_bos, lang, think_eos,   # 7  (+speaker +pad,bos)
//                          spk, pad, bos ]
//   left      = [ tts_pad ]*(7-2) ++ [ tts_bos ]                # 6
//   body0     = left + codec_pfx[:-1]                           # 6
//   txt       = proj_table[ ids[3:-5] ] ++ [ tts_eos ]          # (L+1)   L = body text len
//   blk       = txt + codec_emb[pad]*(L+1)                      # (L+1)
//   tail      = tts_pad + codec_emb[bos]                        # 1
//   inputs_embeds = role ++ body0 ++ blk ++ tail                # 3+6+(L+1)+1
//   trailing_text_hidden = tts_pad                              # 1
//
// where tts_bos/eos/pad = proj_table[tts_bos_id/tts_eos_id/tts_pad_id].
#pragma once

#include <stdint.h>
#include <string>
#include <vector>

namespace qwen3_tts {

// CustomVoice 0.6B special-token ids (from config.json).
struct PrefillIds {
    int hidden = 1024;
    int vocab = 151936;
    int codec_vocab = 3072;
    // tts special (text-side, indexed into proj_table)
    int tts_bos = 151672;
    int tts_eos = 151673;
    int tts_pad = 151671;
    // codec special (indexed into codec_emb)
    int codec_think = 2154;
    int codec_think_bos = 2156;
    int codec_think_eos = 2157;
    int codec_nothink = 2155;
    int codec_pad = 2148;
    int codec_bos = 2149;
    // role prefix / suffix token counts in the chatml id stream
    int role_prefix = 3;  // <|im_start|>assistant\n
    int suffix = 5;       // <|im_end|>\n<|im_start|>assistant\n
};

// Assemble the prefill. token_ids: the full ChatML id stream from the tokenizer.
// proj_table: VOCAB*hidden f32. codec_emb: codec_vocab*hidden f32.
// spk_id: preset-speaker codec id (e.g. serena=3066). lang_id: codec language id
// (e.g. chinese=2055), or -1 for the nothink branch.
// Returns inputs_embeds flattened (S*hidden) and sets S; trailing is hidden-long.
inline std::vector<float> assemble_prefill(
    const std::vector<int>& token_ids,
    const float* proj_table, const float* codec_emb,
    int spk_id, int lang_id, const PrefillIds& P,
    int& S_out, std::vector<float>& trailing_out)
{
    const int H = P.hidden;
    auto proj = [&](int t) { return proj_table + (size_t)t * H; };
    auto codec = [&](int c) { return codec_emb + (size_t)c * H; };

    const float* tts_bos = proj(P.tts_bos);
    const float* tts_eos = proj(P.tts_eos);
    const float* tts_pad = proj(P.tts_pad);

    // codec prefix ids: think branch when a concrete language id is given.
    std::vector<int> pfx_ids;
    if (lang_id >= 0)
        pfx_ids = {P.codec_think, P.codec_think_bos, lang_id, P.codec_think_eos};
    else
        pfx_ids = {P.codec_nothink, P.codec_think_bos, P.codec_think_eos};
    // codec_pfx = [pfx..., spk, pad, bos]
    std::vector<int> codec_pfx = pfx_ids;
    codec_pfx.push_back(spk_id);
    codec_pfx.push_back(P.codec_pad);
    codec_pfx.push_back(P.codec_bos);
    const int n = (int)codec_pfx.size();  // 7 with speaker + chinese

    const int L = (int)token_ids.size() - P.role_prefix - P.suffix;  // body text len
    const int S = P.role_prefix + (n - 1) + (L + 1) + 1;             // total prefill len
    S_out = S;

    std::vector<float> out((size_t)S * H, 0.f);
    float* o = out.data();
    int row = 0;

    // role = proj_table[ ids[0:3] ]
    for (int i = 0; i < P.role_prefix; i++) {
        const float* p = proj(token_ids[i]);
        for (int k = 0; k < H; k++) o[(size_t)row * H + k] = p[k];
        row++;
    }

    // body0 = ([tts_pad]*(n-2) ++ tts_bos) + codec_pfx[:-1]
    for (int i = 0; i < n - 1; i++) {
        const float* left = (i < n - 2) ? tts_pad : tts_bos;
        const float* c = codec(codec_pfx[i]);
        for (int k = 0; k < H; k++) o[(size_t)row * H + k] = left[k] + c[k];
        row++;
    }

    // blk = (proj_table[ ids[3:-5] ] ++ tts_eos) + codec_emb[pad]*(L+1)
    const float* cpad = codec(P.codec_pad);
    for (int i = 0; i < L + 1; i++) {
        const float* txt = (i < L) ? proj(token_ids[P.role_prefix + i]) : tts_eos;
        for (int k = 0; k < H; k++) o[(size_t)row * H + k] = txt[k] + cpad[k];
        row++;
    }

    // tail = tts_pad + codec_emb[bos]
    {
        const float* cbos = codec(P.codec_bos);
        for (int k = 0; k < H; k++) o[(size_t)row * H + k] = tts_pad[k] + cbos[k];
        row++;
    }

    // trailing_text_hidden = tts_pad (1 vector)
    trailing_out.assign(tts_pad, tts_pad + H);
    return out;
}

} // namespace qwen3_tts
