// SPDX-License-Identifier: Apache-2.0
// Qwen3-TTS ncnn inference pipeline: the AR decode loop (Talker outer + Code
// Predictor nested 15-pass + aggregation) and the speech-decoder vocoder chain
// (RVQ gather -> pre_conv -> transformer -> conv tail -> waveform).
//
// Both stages are validated to reproduce the PyTorch reference: the AR loop is
// teacher-force-exact on every frame's cb0, and the vocoder is cosine-sim
// 1.000000 vs the golden waveform. See memory qwen3-tts-cpp-integration.
#pragma once

#include "net.h"
#include "rope.h"
#include "snakebeta.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <vector>

namespace qwen3_tts {

// ---- small raw-binary loaders (tensors dumped as contiguous float32/int32) --
inline bool load_f32(const std::string& path, std::vector<float>& out)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path.c_str()); return false; }
    fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    out.resize(n / sizeof(float));
    size_t got = fread(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    return got == out.size();
}

inline bool load_i32(const std::string& path, std::vector<int>& out)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path.c_str()); return false; }
    fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    out.resize(n / sizeof(int));
    size_t got = fread(out.data(), sizeof(int), out.size(), f);
    fclose(f);
    return got == out.size();
}

// ---- Talker / Code Predictor dimensions (0.6B) ------------------------------
struct TalkerDims {
    static const int HIDDEN = 1024;
    static const int HEAD_DIM = 128;
    static const int CODEC_VOCAB = 3072;
    static const int CP_VOCAB = 2048;
    static const int N_TK = 28;
    static const int N_CP = 5;
    static const int NUM_CODE_GROUPS = 16;
    static const int CODEC_EOS = 2150;
    static constexpr double THETA = 1e6;
};

// additive causal mask (cur rows, past+cur cols): 0 allowed, -inf above diagonal
inline ncnn::Mat causal_mask(int cur, int past)
{
    int dst = past + cur;
    ncnn::Mat m(dst, cur);
    for (int i = 0; i < cur; i++)
    {
        float* r = m.row(i);
        for (int j = 0; j < dst; j++)
            r[j] = (j <= past + i) ? 0.f : -INFINITY;
    }
    return m;
}

// run one decoder pass. embeds (S,HIDDEN) as Mat(w=HIDDEN,h=S). caches_in: NULL
// to skip feeding (prefill), else 2*nlayers Mats. Returns hidden (w=HIDDEN,h=S),
// fills caches_out (2*nlayers).
inline ncnn::Mat run_decoder(ncnn::Net& net, const ncnn::Mat& embeds,
                             const ncnn::Mat& cos_c, const ncnn::Mat& sin_c,
                             const ncnn::Mat& mask, std::vector<ncnn::Mat>* caches_in,
                             int nlayers, std::vector<ncnn::Mat>& caches_out)
{
    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", embeds);
    ex.input("in1", mask);
    ex.input("in2", cos_c);
    ex.input("in3", sin_c);
    if (caches_in)
    {
        for (int i = 0; i < nlayers; i++)
        {
            char nm[32];
            sprintf(nm, "cache_k%d", i); ex.input(nm, (*caches_in)[2 * i]);
            sprintf(nm, "cache_v%d", i); ex.input(nm, (*caches_in)[2 * i + 1]);
        }
    }
    ncnn::Mat hidden;
    ex.extract("out0", hidden);
    caches_out.resize(2 * nlayers);
    for (int i = 0; i < nlayers; i++)
    {
        char nm[32];
        sprintf(nm, "out_cache_k%d", i); ex.extract(nm, caches_out[2 * i]);
        sprintf(nm, "out_cache_v%d", i); ex.extract(nm, caches_out[2 * i + 1]);
    }
    return hidden;
}

// argmax of W(rows=vocab, HIDDEN) @ vec(HIDDEN), with optional suppress list
inline int argmax_head(const float* W, int vocab, const float* vec,
                       const int* suppress, int nsup)
{
    const int H = TalkerDims::HIDDEN;
    std::vector<float> logits(vocab);
    for (int r = 0; r < vocab; r++)
    {
        const float* wr = W + (size_t)r * H;
        double s = 0;
        for (int k = 0; k < H; k++) s += (double)wr[k] * vec[k];
        logits[r] = (float)s;
    }
    for (int i = 0; i < nsup; i++) logits[suppress[i]] = -INFINITY;
    int best = 0; float bv = logits[0];
    for (int r = 1; r < vocab; r++) if (logits[r] > bv) { bv = logits[r]; best = r; }
    return best;
}

// ---- weights + nets for the AR loop -----------------------------------------
struct TalkerRuntime {
    ncnn::Net tk_net, cp_net;
    std::vector<float> tk_emb;    // (3072,1024) talker codec_embedding
    std::vector<float> cp_emb;    // (15,2048,1024) code_predictor codec_embedding
    std::vector<float> codec_head;// (3072,1024)
    std::vector<float> cp_head;   // (15,2048,1024)
    std::vector<int> suppress;

    bool load(const std::string& P, const std::string& D)
    {
        ncnn::Option opt;
        opt.use_vulkan_compute = false;
        opt.use_fp16_packed = false; opt.use_fp16_storage = false; opt.use_fp16_arithmetic = false;
        opt.num_threads = 4;
        tk_net.opt = opt; cp_net.opt = opt;
        if (tk_net.load_param((P + "/talker_decoder_kvcache.ncnn.param").c_str())) return false;
        if (tk_net.load_model((P + "/talker_decoder_kvcache.ncnn.bin").c_str())) return false;
        if (cp_net.load_param((P + "/code_predictor_kvcache.ncnn.param").c_str())) return false;
        if (cp_net.load_model((P + "/code_predictor_kvcache.ncnn.bin").c_str())) return false;
        if (!load_f32(D + "/talker_codec_embedding.f32", tk_emb)) return false;
        if (!load_f32(D + "/cp_codec_embedding.f32", cp_emb)) return false;
        if (!load_f32(D + "/talker_codec_head_w.f32", codec_head)) return false;
        if (!load_f32(D + "/cp_lm_head_w.f32", cp_head)) return false;
        for (int i = 2048; i < 3072; i++) if (i != TalkerDims::CODEC_EOS) suppress.push_back(i);
        return true;
    }
};

// Run the full AR loop from an assembled prefill. Produces codes [T,16].
// prefill_embeds: S*HIDDEN, trailing: N_TEXT*HIDDEN, pad: HIDDEN.
inline std::vector<std::vector<int> > run_ar_loop(
    TalkerRuntime& rt, const float* prefill_embeds, int S,
    const float* trailing, int N_TEXT, const float* pad, int max_frames)
{
    typedef TalkerDims D;
    const int H = D::HIDDEN;

    ncnn::Mat pref(H, S);
    memcpy(pref, prefill_embeds, (size_t)S * H * sizeof(float));
    ncnn::Mat cos_c, sin_c;
    build_rope_cache(S, D::HEAD_DIM, D::THETA, 0, cos_c, sin_c);
    ncnn::Mat mask = causal_mask(S, 0);
    std::vector<ncnn::Mat> tk_caches;
    ncnn::Mat hidden = run_decoder(rt.tk_net, pref, cos_c, sin_c, mask, NULL, D::N_TK, tk_caches);

    std::vector<float> past_hidden(H);
    memcpy(past_hidden.data(), hidden.row(S - 1), H * sizeof(float));
    int cb0 = argmax_head(rt.codec_head.data(), D::CODEC_VOCAB, past_hidden.data(),
                          rt.suppress.data(), (int)rt.suppress.size());

    std::vector<std::vector<int> > frames;
    int past_len = S, step = 0;
    while (cb0 != D::CODEC_EOS && step < max_frames)
    {
        // Code Predictor 15-pass inner loop
        std::vector<int> cp_codes;
        ncnn::Mat cp_in(H, 2);
        memcpy(cp_in.row(0), past_hidden.data(), H * sizeof(float));
        memcpy(cp_in.row(1), &rt.tk_emb[(size_t)cb0 * H], H * sizeof(float));
        ncnn::Mat cc, sc; build_rope_cache(2, D::HEAD_DIM, D::THETA, 0, cc, sc);
        ncnn::Mat cpm = causal_mask(2, 0);
        std::vector<ncnn::Mat> cp_caches;
        ncnn::Mat cph = run_decoder(rt.cp_net, cp_in, cc, sc, cpm, NULL, D::N_CP, cp_caches);
        int cb = argmax_head(&rt.cp_head[0], D::CP_VOCAB, cph.row(1), NULL, 0);
        cp_codes.push_back(cb);
        for (int g = 1; g < 15; g++)
        {
            ncnn::Mat emb(H, 1);
            memcpy(emb.row(0), &rt.cp_emb[((size_t)(g - 1) * D::CP_VOCAB + cb) * H], H * sizeof(float));
            int pos = 1 + g;
            ncnn::Mat cc1, sc1; build_rope_cache(1, D::HEAD_DIM, D::THETA, pos, cc1, sc1);
            ncnn::Mat m1 = causal_mask(1, pos);
            std::vector<ncnn::Mat> cp_caches2;
            ncnn::Mat h1 = run_decoder(rt.cp_net, emb, cc1, sc1, m1, &cp_caches, D::N_CP, cp_caches2);
            cb = argmax_head(&rt.cp_head[(size_t)g * D::CP_VOCAB * H], D::CP_VOCAB, h1.row(0), NULL, 0);
            cp_codes.push_back(cb);
            cp_caches = cp_caches2;
        }

        std::vector<int> frame; frame.push_back(cb0);
        for (int i = 0; i < 15; i++) frame.push_back(cp_codes[i]);
        frames.push_back(frame);

        // aggregate: sum 16 code embeds + trailing text (or pad)
        std::vector<double> agg(H, 0.0);
        const float* e0 = &rt.tk_emb[(size_t)cb0 * H];
        for (int k = 0; k < H; k++) agg[k] = e0[k];
        for (int i = 0; i < 15; i++)
        {
            const float* ei = &rt.cp_emb[((size_t)i * D::CP_VOCAB + cp_codes[i]) * H];
            for (int k = 0; k < H; k++) agg[k] += ei[k];
        }
        const float* text = (step < N_TEXT) ? &trailing[(size_t)step * H] : pad;
        ncnn::Mat step_emb(H, 1);
        float* sep = step_emb.row(0);
        for (int k = 0; k < H; k++) sep[k] = (float)agg[k] + text[k];

        // Talker decode step
        ncnn::Mat cc2, sc2; build_rope_cache(1, D::HEAD_DIM, D::THETA, past_len, cc2, sc2);
        ncnn::Mat m2 = causal_mask(1, past_len);
        std::vector<ncnn::Mat> tk_caches2;
        hidden = run_decoder(rt.tk_net, step_emb, cc2, sc2, m2, &tk_caches, D::N_TK, tk_caches2);
        tk_caches = tk_caches2;
        memcpy(past_hidden.data(), hidden.row(0), H * sizeof(float));
        cb0 = argmax_head(rt.codec_head.data(), D::CODEC_VOCAB, past_hidden.data(),
                          rt.suppress.data(), (int)rt.suppress.size());
        past_len++; step++;
    }
    return frames;
}

// ---- vocoder: codes [T,16] -> waveform --------------------------------------
struct VocoderRuntime {
    std::string dir;
    std::vector<float> rvq;   // 16 x (2048,512)
    ncnn::Option opt;

    bool load(const std::string& P)
    {
        dir = P;
        opt.use_vulkan_compute = false;
        opt.use_fp16_packed = false; opt.use_fp16_storage = false; opt.use_fp16_arithmetic = false;
        opt.num_threads = 4;
        if (!load_f32(P + "/rvq_tables.bin", rvq)) return false;
        return (int)rvq.size() == 16 * 2048 * 512;
    }

    // codes [T][16] -> mono float waveform (T*1920 samples)
    std::vector<float> decode(const std::vector<std::vector<int> >& codes)
    {
        const int T = (int)codes.size();
        const int BINS = 2048, DIM = 512, HIDDEN = 1024;

        // RVQ gather+sum -> (w=T, h=1, c=512)
        ncnn::Mat rvq_out(T, 1, DIM);
        rvq_out.fill(0.f);
        for (int t = 0; t < T; t++)
            for (int q = 0; q < 16; q++)
            {
                int code = codes[t][q]; if (code < 0) code = 0;
                const float* tbl = &rvq[(size_t)q * BINS * DIM + (size_t)code * DIM];
                for (int c = 0; c < DIM; c++) rvq_out.channel(c)[t] += tbl[c];
            }

        // pre_conv (512->1024)
        ncnn::Mat pc;
        { ncnn::Net net; net.opt = opt;
          net.load_param((dir + "/dec_pre_conv.ncnn.param").c_str());
          net.load_model((dir + "/dec_pre_conv.ncnn.bin").c_str());
          ncnn::Extractor ex = net.create_extractor();
          ex.input("in0", rvq_out); ex.extract("out0", pc); }

        // transpose (w=T,h=1,c=1024) -> transformer 2D (w=1024,h=T)
        ncnn::Mat tf_in(HIDDEN, T);
        for (int t = 0; t < T; t++)
            for (int c = 0; c < HIDDEN; c++) tf_in.row(t)[c] = pc.channel(c)[t];

        // decoder transformer (theta 1e4, head_dim 64, sliding window 72)
        ncnn::Mat cos_c, sin_c;
        build_rope_cache(T, 64, 10000.0, 0, cos_c, sin_c);
        ncnn::Mat mask(T, T);
        for (int i = 0; i < T; i++) { float* r = mask.row(i);
            for (int j = 0; j < T; j++) { int lo = i - 72 + 1; if (lo < 0) lo = 0;
                r[j] = (j >= lo && j <= i) ? 0.f : -INFINITY; } }
        ncnn::Mat tf_out;
        { ncnn::Net net; net.opt = opt;
          net.load_param((dir + "/decoder_transformer.ncnn.param").c_str());
          net.load_model((dir + "/decoder_transformer.ncnn.bin").c_str());
          ncnn::Extractor ex = net.create_extractor();
          ex.input("in0", tf_in); ex.input("in1", cos_c); ex.input("in2", sin_c); ex.input("in3", mask);
          ex.extract("out0", tf_out); }

        // transpose back to conv 3D (w=T,h=1,c=1024)
        ncnn::Mat tail_in(T, 1, HIDDEN);
        for (int c = 0; c < HIDDEN; c++)
            for (int t = 0; t < T; t++) tail_in.channel(c)[t] = tf_out.row(t)[c];

        // conv tail (decomposed SnakeBeta -> native ops; custom layer as fallback)
        ncnn::Mat wav;
        { ncnn::Net net; net.opt = opt;
          net.register_custom_layer(
              "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2.SnakeBeta",
              SnakeBeta_layer_creator);
          net.load_param((dir + "/dec_conv_tail.ncnn.param").c_str());
          net.load_model((dir + "/dec_conv_tail.ncnn.bin").c_str());
          ncnn::Extractor ex = net.create_extractor();
          ex.input("in0", tail_in); ex.extract("out0", wav); }

        std::vector<float> out(wav.w);
        memcpy(out.data(), (const float*)wav, wav.w * sizeof(float));
        return out;
    }
};

} // namespace qwen3_tts
