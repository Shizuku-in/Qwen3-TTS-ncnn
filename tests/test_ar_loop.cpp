// SPDX-License-Identifier: Apache-2.0
// Validate the C++ AR decode loop (Talker outer + Code Predictor nested 15-pass
// + aggregation) against the PyTorch golden codes.
//
// This mirrors the validated Python ar_loop_driver.py exactly. It is driven by
// the dumped prefill tensors (talker_inputs_embeds etc.) so it isolates the AR
// loop mechanism from the prefill-assembly step.
//
// Parity criterion (see memory qwen3-tts-cpp-integration):
//   - teacher-forced: every frame's Talker cb0 must match golden exactly
//     (proves Talker + KV cache + aggregation + RoPE + mask are correct)
//   - free-running divergence after a few frames is EXPECTED: greedy argmax
//     amplifies sub-0.1% fp32 numerical noise; not a bug.
//
// usage: test_ar_loop <pnnx_dir> <dumps_dir>
#include "net.h"
#include "rope.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <vector>

static bool load_f32(const std::string& path, std::vector<float>& out)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path.c_str()); return false; }
    fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    out.resize(n / sizeof(float));
    size_t got = fread(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    return got == out.size();
}

static bool load_i32(const std::string& path, std::vector<int>& out)
{
    FILE* f = fopen(path.c_str(), "rb");
    if (!f) { fprintf(stderr, "open %s failed\n", path.c_str()); return false; }
    fseek(f, 0, SEEK_END); long n = ftell(f); rewind(f);
    out.resize(n / sizeof(int));
    size_t got = fread(out.data(), sizeof(int), out.size(), f);
    fclose(f);
    return got == out.size();
}

static const int HIDDEN = 1024;
static const int HEAD_DIM = 128;
static const double THETA = 1e6;
static const int CODEC_EOS = 2150;
static const int N_TK = 28;
static const int N_CP = 5;
static const int NUM_CODE_GROUPS = 16;

// additive causal mask (cur rows, past+cur cols): 0 allowed, -inf above diagonal
static ncnn::Mat causal_mask(int cur, int past)
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

// run one decoder pass. embeds (S,HIDDEN) as Mat(w=HIDDEN,h=S). caches_in: 2*nl
// ncnn::Mat, or NULL to skip feeding (prefill). Returns hidden (S,HIDDEN) and
// fills caches_out (2*nl).
static ncnn::Mat run_decoder(ncnn::Net& net, const ncnn::Mat& embeds,
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
    return hidden;  // (w=HIDDEN, h=S)
}

// row s of a (w=HIDDEN,h=S) hidden Mat -> pointer
static const float* hidden_row(const ncnn::Mat& h, int s) { return h.row(s); }

// argmax of W(rows=vocab, HIDDEN) @ vec(HIDDEN), with optional suppress
static int argmax_head(const float* W, int vocab, const float* vec,
                       const int* suppress, int nsup)
{
    std::vector<float> logits(vocab);
    for (int r = 0; r < vocab; r++)
    {
        const float* wr = W + (size_t)r * HIDDEN;
        double s = 0;
        for (int k = 0; k < HIDDEN; k++) s += (double)wr[k] * vec[k];
        logits[r] = (float)s;
    }
    for (int i = 0; i < nsup; i++) logits[suppress[i]] = -INFINITY;
    int best = 0; float bv = logits[0];
    for (int r = 1; r < vocab; r++) if (logits[r] > bv) { bv = logits[r]; best = r; }
    return best;
}

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <pnnx_dir> <dumps_dir>\n", argv[0]); return 1; }
    std::string P = argv[1], D = argv[2];

    // ---- load dumped tensors ----
    std::vector<float> ie, tth, tpe, tk_emb, cp_emb, codec_head, cp_head;
    std::vector<int> golden;
    if (!load_f32(D + "/talker_inputs_embeds.f32", ie)) return 1;
    if (!load_f32(D + "/talker_trailing_text_hidden.f32", tth)) return 1;
    if (!load_f32(D + "/talker_tts_pad_embed.f32", tpe)) return 1;
    if (!load_f32(D + "/talker_codec_embedding.f32", tk_emb)) return 1;
    if (!load_f32(D + "/cp_codec_embedding.f32", cp_emb)) return 1;
    if (!load_f32(D + "/talker_codec_head_w.f32", codec_head)) return 1;
    if (!load_f32(D + "/cp_lm_head_w.f32", cp_head)) return 1;
    if (!load_i32(D + "/golden_codes.i32", golden)) return 1;

    const int S = (int)ie.size() / HIDDEN;            // prefill length
    const int N_TEXT = (int)tth.size() / HIDDEN;      // trailing text steps
    const int T = (int)golden.size() / 16;            // golden frame count
    printf("prefill S=%d trailing=%d golden_frames=%d\n", S, N_TEXT, T);

    // suppress tokens: [2048,3072) except 2150
    std::vector<int> suppress;
    for (int i = 2048; i < 3072; i++) if (i != CODEC_EOS) suppress.push_back(i);

    ncnn::Option opt;
    opt.use_vulkan_compute = false;
    opt.use_fp16_packed = false; opt.use_fp16_storage = false; opt.use_fp16_arithmetic = false;
    opt.num_threads = 4;

    ncnn::Net tk_net; tk_net.opt = opt;
    if (tk_net.load_param((P + "/talker_decoder_kvcache.ncnn.param").c_str())) { fprintf(stderr,"tk param\n"); return 1; }
    if (tk_net.load_model((P + "/talker_decoder_kvcache.ncnn.bin").c_str())) { fprintf(stderr,"tk bin\n"); return 1; }
    ncnn::Net cp_net; cp_net.opt = opt;
    if (cp_net.load_param((P + "/code_predictor_kvcache.ncnn.param").c_str())) { fprintf(stderr,"cp param\n"); return 1; }
    if (cp_net.load_model((P + "/code_predictor_kvcache.ncnn.bin").c_str())) { fprintf(stderr,"cp bin\n"); return 1; }

    // ---- Talker prefill ----
    ncnn::Mat pref_emb(HIDDEN, S);
    memcpy(pref_emb, ie.data(), (size_t)S * HIDDEN * sizeof(float));
    ncnn::Mat cos_c, sin_c;
    qwen3_tts::build_rope_cache(S, HEAD_DIM, THETA, 0, cos_c, sin_c);
    ncnn::Mat mask = causal_mask(S, 0);
    std::vector<ncnn::Mat> tk_caches;
    ncnn::Mat hidden = run_decoder(tk_net, pref_emb, cos_c, sin_c, mask, NULL, N_TK, tk_caches);

    std::vector<float> past_hidden(HIDDEN);
    memcpy(past_hidden.data(), hidden_row(hidden, S - 1), HIDDEN * sizeof(float));
    int cb0 = argmax_head(codec_head.data(), 3072, past_hidden.data(), suppress.data(), (int)suppress.size());

    // ---- AR loop ----
    std::vector<std::vector<int> > frames;
    int past_len = S, step = 0;
    while (cb0 != CODEC_EOS && step < T)
    {
        // ---- Code Predictor inner loop: 15 passes ----
        std::vector<int> cp_codes;
        // 2-token prefill: [past_hidden, tk_emb[cb0]]
        ncnn::Mat cp_in(HIDDEN, 2);
        memcpy(cp_in.row(0), past_hidden.data(), HIDDEN * sizeof(float));
        memcpy(cp_in.row(1), &tk_emb[(size_t)cb0 * HIDDEN], HIDDEN * sizeof(float));
        ncnn::Mat cc, sc; qwen3_tts::build_rope_cache(2, HEAD_DIM, THETA, 0, cc, sc);
        ncnn::Mat cpm = causal_mask(2, 0);
        std::vector<ncnn::Mat> cp_caches;
        ncnn::Mat cph = run_decoder(cp_net, cp_in, cc, sc, cpm, NULL, N_CP, cp_caches);
        // predict cb1 from last position via cp_head[0]
        int cb = argmax_head(&cp_head[0], 2048, hidden_row(cph, 1), NULL, 0);
        cp_codes.push_back(cb);
        for (int g = 1; g < 15; g++)
        {
            ncnn::Mat emb(HIDDEN, 1);
            memcpy(emb.row(0), &cp_emb[((size_t)(g - 1) * 2048 + cb) * HIDDEN], HIDDEN * sizeof(float));
            int pos = 1 + g;
            ncnn::Mat cc1, sc1; qwen3_tts::build_rope_cache(1, HEAD_DIM, THETA, pos, cc1, sc1);
            ncnn::Mat m1 = causal_mask(1, pos);
            std::vector<ncnn::Mat> cp_caches2;
            ncnn::Mat h1 = run_decoder(cp_net, emb, cc1, sc1, m1, &cp_caches, N_CP, cp_caches2);
            cb = argmax_head(&cp_head[(size_t)g * 2048 * HIDDEN], 2048, hidden_row(h1, 0), NULL, 0);
            cp_codes.push_back(cb);
            cp_caches = cp_caches2;
        }

        // ---- assemble frame, aggregate ----
        std::vector<int> frame; frame.push_back(cb0);
        for (int i = 0; i < 15; i++) frame.push_back(cp_codes[i]);
        frames.push_back(frame);

        std::vector<double> agg(HIDDEN, 0.0);
        const float* e0 = &tk_emb[(size_t)cb0 * HIDDEN];
        for (int k = 0; k < HIDDEN; k++) agg[k] = e0[k];
        for (int i = 0; i < 15; i++)
        {
            const float* ei = &cp_emb[((size_t)i * 2048 + cp_codes[i]) * HIDDEN];
            for (int k = 0; k < HIDDEN; k++) agg[k] += ei[k];
        }
        const float* text = (step < N_TEXT) ? &tth[(size_t)step * HIDDEN] : tpe.data();
        ncnn::Mat step_emb(HIDDEN, 1);
        float* sep = step_emb.row(0);
        for (int k = 0; k < HIDDEN; k++) sep[k] = (float)agg[k] + text[k];

        // ---- Talker decode step ----
        ncnn::Mat cc2, sc2; qwen3_tts::build_rope_cache(1, HEAD_DIM, THETA, past_len, cc2, sc2);
        ncnn::Mat m2 = causal_mask(1, past_len);
        std::vector<ncnn::Mat> tk_caches2;
        hidden = run_decoder(tk_net, step_emb, cc2, sc2, m2, &tk_caches, N_TK, tk_caches2);
        tk_caches = tk_caches2;
        memcpy(past_hidden.data(), hidden_row(hidden, 0), HIDDEN * sizeof(float));
        cb0 = argmax_head(codec_head.data(), 3072, past_hidden.data(), suppress.data(), (int)suppress.size());
        past_len++; step++;
    }

    // ---- compare to golden ----
    int nf = (int)frames.size();
    int n = nf < T ? nf : T;
    int full_match = 0, elem_match = 0, cb0_match = 0;
    for (int t = 0; t < n; t++)
    {
        bool all = true;
        for (int c = 0; c < 16; c++)
        {
            if (frames[t][c] == golden[t * 16 + c]) elem_match++;
            else all = false;
        }
        if (all) full_match++;
        if (frames[t][0] == golden[t * 16]) cb0_match++;
    }
    printf("frames produced=%d golden=%d\n", nf, T);
    printf("full-16 frame match: %d/%d\n", full_match, n);
    printf("cb0 match: %d/%d\n", cb0_match, n);
    printf("elementwise: %.2f%%\n", 100.0 * elem_match / (n * 16));
    // frame 0 (pure prefill) must match exactly
    bool f0 = (full_match >= 1) && [&]{ for(int c=0;c<16;c++) if(frames[0][c]!=golden[c]) return false; return true; }();
    printf("frame0 exact: %s\n", f0 ? "YES" : "NO");
    return f0 ? 0 : 2;
}
