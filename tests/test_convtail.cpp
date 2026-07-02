// SPDX-License-Identifier: Apache-2.0
// Validate the converted speech-decoder conv tail (pre_conv + upsample +
// decoder blocks + SnakeBeta) against the PyTorch golden output.
//
// This is the first C++ checkpoint: it exercises the CMake/ncnn link AND the
// custom SnakeBeta layer together.
//
// usage: test_convtail <param> <bin> <in.bin> <gold.bin>
#include "net.h"
#include "snakebeta.h"

#include <stdio.h>
#include <string.h>
#include <math.h>
#include <vector>

static bool load_bin(const char* path, std::vector<float>& out)
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

int main(int argc, char** argv)
{
    if (argc < 5)
    {
        fprintf(stderr, "usage: %s <param> <bin> <in.bin> <gold.bin>\n", argv[0]);
        return 1;
    }
    const char* param = argv[1];
    const char* binf = argv[2];
    const char* inf = argv[3];
    const char* goldf = argv[4];

    ncnn::Net net;
    net.opt.use_vulkan_compute = false;
    net.opt.use_fp16_packed = false;
    net.opt.use_fp16_storage = false;
    net.opt.use_fp16_arithmetic = false;
    net.register_custom_layer(
        "qwen_tts.core.tokenizer_12hz.modeling_qwen3_tts_tokenizer_v2.SnakeBeta",
        qwen3_tts::SnakeBeta_layer_creator);

    if (net.load_param(param) != 0) { fprintf(stderr, "load_param failed\n"); return 1; }
    if (net.load_model(binf) != 0) { fprintf(stderr, "load_model failed\n"); return 1; }

    std::vector<float> input, gold;
    if (!load_bin(inf, input)) { fprintf(stderr, "load input failed\n"); return 1; }
    if (!load_bin(goldf, gold)) { fprintf(stderr, "load gold failed\n"); return 1; }

    // input was traced as (1,1024,8)=(batch,channel,length); ncnn drops batch.
    // The ncnn.py harness feeds ncnn.Mat((1024,8) numpy) -> Mat(w=8, h=1024).
    // Convolution1D reads h as input channels, w as length.
    ncnn::Mat in(8, 1024);
    memcpy(in, input.data(), 1024 * 8 * sizeof(float));

    fprintf(stderr, "in dims=%d w=%d h=%d d=%d c=%d\n", in.dims, in.w, in.h, in.d, in.c);

    ncnn::Extractor ex = net.create_extractor();
    ex.input("in0", in);
    ncnn::Mat out;
    ex.extract("out0", out);

    fprintf(stderr, "out dims=%d w=%d h=%d d=%d c=%d\n", out.dims, out.w, out.h, out.d, out.c);

    const int n = (int)gold.size();
    const int on = out.w * out.h * out.d * out.c;
    if (on != n) fprintf(stderr, "WARN size mismatch: ncnn=%d gold=%d\n", on, n);

    double max_abs = 0, dot = 0, gg = 0, oo = 0;
    const float* op = out;
    for (int i = 0; i < n && i < on; i++)
    {
        double d = fabs(op[i] - gold[i]);
        if (d > max_abs) max_abs = d;
        dot += (double)op[i] * gold[i];
        gg += (double)gold[i] * gold[i];
        oo += (double)op[i] * op[i];
    }
    double cos = dot / (sqrt(gg) * sqrt(oo) + 1e-12);
    printf("max_abs=%.6e cos_sim=%.6f (ncnn=%d gold=%d)\n", max_abs, cos, on, n);
    return (max_abs < 1e-2 && cos > 0.9999) ? 0 : 2;
}
