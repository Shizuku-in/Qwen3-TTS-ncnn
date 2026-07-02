// SPDX-License-Identifier: Apache-2.0
// Full speech-decoder (vocoder) chain in C++: codes -> 24kHz waveform.
//
//   codes [T,16] -> RVQ gather+sum (512,T) -> pre_conv (1024,T)
//     -> decoder_transformer (8 layers) -> conv_tail (upsample+decoder)
//     -> wav [1920*T]
//
// Validated against the P0 golden wav (golden_out.wav) for the same codes.
//
// Layout notes (hard-won, see memory qwen3-tts-cpp-integration):
//   - ncnn Convolution1D nets need 3D input (w=length, h=1, c=channels).
//   - transformer nets (pnnx from (B,T,H)) take 2D input (w=H, h=T).
//   - between conv and transformer stages we transpose explicitly.
//
// usage: test_vocoder <artifact_dir> <codes.txt> <gold_wav_f32.bin>
#include "net.h"
#include "snakebeta.h"
#include "rope.h"

#include <stdio.h>
#include <stdlib.h>
#include <string.h>
#include <math.h>
#include <string>
#include <vector>

static bool load_f32(const char* path, std::vector<float>& out)
{
    FILE* f = fopen(path, "rb");
    if (!f) return false;
    fseek(f, 0, SEEK_END);
    long n = ftell(f);
    rewind(f);
    out.resize(n / sizeof(float));
    size_t got = fread(out.data(), sizeof(float), out.size(), f);
    fclose(f);
    return got == out.size();
}

// load golden codes: text file, one frame per line, 16 ints per line
static bool load_codes(const char* path, std::vector<std::vector<int> >& codes)
{
    FILE* f = fopen(path, "r");
    if (!f) return false;
    char line[4096];
    while (fgets(line, sizeof(line), f))
    {
        if (line[0] == '#') continue;
        std::vector<int> row;
        char* p = line;
        while (*p)
        {
            while (*p == ' ' || *p == '\t' || *p == '\n') p++;
            if (!*p) break;
            row.push_back((int)strtol(p, &p, 10));
        }
        if (row.size() >= 16) codes.push_back(std::vector<int>(row.begin(), row.begin() + 16));
    }
    fclose(f);
    return !codes.empty();
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <artifact_dir> <codes.txt> <gold_wav_f32.bin>\n", argv[0]);
        return 1;
    }
    std::string dir = argv[1];
    const char* codes_path = argv[2];
    const char* gold_path = argv[3];

    // ---- load codes ----
    std::vector<std::vector<int> > codes;
    if (!load_codes(codes_path, codes)) { fprintf(stderr, "load codes failed\n"); return 1; }
    const int T = (int)codes.size();
    printf("frames T=%d\n", T);

    // ---- load RVQ tables: 16 x (2048,512) f32 ----
    std::vector<float> rvq;
    if (!load_f32((dir + "/rvq_tables.bin").c_str(), rvq)) { fprintf(stderr, "load rvq failed\n"); return 1; }
    const int BINS = 2048, DIM = 512;
    if ((int)rvq.size() != 16 * BINS * DIM) { fprintf(stderr, "rvq size %zu != %d\n", rvq.size(), 16*BINS*DIM); return 1; }

    // ---- RVQ dequant: hidden(c=512, t) = sem[code0] + sum_{q=1..15} aco[q-1][code_q] ----
    // build pre_conv input directly: 3D (w=T, h=1, c=512)
    ncnn::Mat rvq_out(T, 1, DIM);
    rvq_out.fill(0.f);
    for (int t = 0; t < T; t++)
    {
        for (int q = 0; q < 16; q++)
        {
            int code = codes[t][q];
            if (code < 0) code = 0;
            const float* tbl = &rvq[(size_t)q * BINS * DIM + (size_t)code * DIM];
            for (int c = 0; c < DIM; c++)
                rvq_out.channel(c)[t] += tbl[c];
        }
    }

    ncnn::Option opt;
    opt.use_vulkan_compute = false;
    opt.use_fp16_packed = false;
    opt.use_fp16_storage = false;
    opt.use_fp16_arithmetic = false;
    opt.num_threads = 4;

    // ---- pre_conv: (512,T) -> (1024,T) ----
    ncnn::Mat preconv_out;
    {
        ncnn::Net net; net.opt = opt;
        if (net.load_param((dir + "/dec_pre_conv.ncnn.param").c_str()) != 0) { fprintf(stderr, "pre_conv param\n"); return 1; }
        if (net.load_model((dir + "/dec_pre_conv.ncnn.bin").c_str()) != 0) { fprintf(stderr, "pre_conv bin\n"); return 1; }
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", rvq_out);
        ex.extract("out0", preconv_out);
    }
    printf("pre_conv out: w=%d h=%d c=%d\n", preconv_out.w, preconv_out.h, preconv_out.c);
    // expect (w=T, h=1, c=1024)

    // ---- transpose (w=T,h=1,c=1024) -> transformer input 2D (w=1024, h=T) ----
    const int H = 1024;
    ncnn::Mat tf_in(H, T);
    for (int t = 0; t < T; t++)
        for (int c = 0; c < H; c++)
            tf_in.row(t)[c] = preconv_out.channel(c)[t];

    // ---- decoder transformer: reduced RoPE (theta 1e4, head_dim 64) + sliding-window mask (72) ----
    const int head_dim = 64, window = 72;
    ncnn::Mat cos_c, sin_c;
    qwen3_tts::build_rope_cache(T, head_dim, 10000.0, 0, cos_c, sin_c);
    ncnn::Mat mask(T, T);
    for (int i = 0; i < T; i++)
    {
        float* r = mask.row(i);
        for (int j = 0; j < T; j++)
        {
            int lo = i - window + 1; if (lo < 0) lo = 0;
            r[j] = (j >= lo && j <= i) ? 0.f : -INFINITY;
        }
    }

    ncnn::Mat tf_out;
    {
        ncnn::Net net; net.opt = opt;
        if (net.load_param((dir + "/decoder_transformer.ncnn.param").c_str()) != 0) { fprintf(stderr, "tf param\n"); return 1; }
        if (net.load_model((dir + "/decoder_transformer.ncnn.bin").c_str()) != 0) { fprintf(stderr, "tf bin\n"); return 1; }
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", tf_in);
        ex.input("in1", cos_c);
        ex.input("in2", sin_c);
        ex.input("in3", mask);
        ex.extract("out0", tf_out);
    }
    printf("tf out: w=%d h=%d c=%d\n", tf_out.w, tf_out.h, tf_out.c);
    // expect (w=1024, h=T)

    // ---- transpose (w=1024,h=T) -> conv_tail input 3D (w=T, h=1, c=1024) ----
    ncnn::Mat tail_in(T, 1, H);
    for (int c = 0; c < H; c++)
        for (int t = 0; t < T; t++)
            tail_in.channel(c)[t] = tf_out.row(t)[c];

    // ---- conv tail (upsample + decoder + SnakeBeta) -> wav ----
    ncnn::Mat wav;
    {
        ncnn::Net net; net.opt = opt;
        net.register_custom_layer(
            "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2.SnakeBeta",
            qwen3_tts::SnakeBeta_layer_creator);
        if (net.load_param((dir + "/dec_conv_tail_mo.ncnn.param").c_str()) != 0) { fprintf(stderr, "tail param\n"); return 1; }
        if (net.load_model((dir + "/dec_conv_tail_mo.ncnn.bin").c_str()) != 0) { fprintf(stderr, "tail bin\n"); return 1; }
        ncnn::Extractor ex = net.create_extractor();
        ex.input("in0", tail_in);
        ex.extract("out0", wav);
    }
    const int nwav = wav.w * wav.h * wav.d * wav.c;
    printf("wav: w=%d h=%d c=%d total=%d (expect %d)\n", wav.w, wav.h, wav.c, nwav, T * 1920);

    // ---- compare to golden ----
    std::vector<float> gold;
    if (!load_f32(gold_path, gold)) { fprintf(stderr, "load gold failed\n"); return 1; }
    const int n = (int)gold.size() < nwav ? (int)gold.size() : nwav;
    if ((int)gold.size() != nwav) fprintf(stderr, "WARN size: ncnn=%d gold=%zu\n", nwav, gold.size());

    const float* wp = wav;
    double max_abs = 0, dot = 0, gg = 0, oo = 0;
    for (int i = 0; i < n; i++)
    {
        double d = fabs(wp[i] - gold[i]);
        if (d > max_abs) max_abs = d;
        dot += (double)wp[i] * gold[i];
        gg += (double)gold[i] * gold[i];
        oo += (double)wp[i] * wp[i];
    }
    double cos = dot / (sqrt(gg) * sqrt(oo) + 1e-12);
    printf("max_abs=%.6e cos_sim=%.6f\n", max_abs, cos);
    return (cos > 0.999) ? 0 : 2;
}
