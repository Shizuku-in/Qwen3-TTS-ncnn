// SPDX-License-Identifier: Apache-2.0
// Qwen3-TTS ncnn synthesizer (end-to-end from an assembled prefill).
//
// Chains the two validated stages:
//   assembled prefill embeds -> AR loop (Talker + Code Predictor) -> codes [T,16]
//     -> vocoder (RVQ -> pre_conv -> transformer -> conv tail) -> 24 kHz WAV
//
// Prefill assembly (text -> embeds) is model-type-specific and still done in the
// PyTorch reference; here we consume the dumped prefill tensors. The AR loop and
// vocoder are pure C++/ncnn.
//
// usage: qwen3_tts <pnnx_dir> <dumps_dir> <out.wav> [max_frames]
#include "pipeline.h"
#include "wav.h"

#include <stdio.h>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <pnnx_dir> <dumps_dir> <out.wav> [max_frames]\n", argv[0]);
        return 1;
    }
    const std::string P = argv[1];
    const std::string D = argv[2];
    const char* out_wav = argv[3];
    const int max_frames = (argc > 4) ? atoi(argv[4]) : 256;

    using namespace qwen3_tts;
    typedef TalkerDims TD;
    const int H = TD::HIDDEN;

    // ---- load AR-loop runtime (nets + weights) ----
    TalkerRuntime rt;
    if (!rt.load(P, D)) { fprintf(stderr, "TalkerRuntime load failed\n"); return 1; }

    // ---- load the assembled prefill tensors ----
    std::vector<float> ie, tth, tpe;
    if (!load_f32(D + "/talker_inputs_embeds.f32", ie)) return 1;
    if (!load_f32(D + "/talker_trailing_text_hidden.f32", tth)) return 1;
    if (!load_f32(D + "/talker_tts_pad_embed.f32", tpe)) return 1;
    const int S = (int)ie.size() / H;
    const int N_TEXT = (int)tth.size() / H;
    printf("[qwen3_tts] prefill S=%d trailing=%d\n", S, N_TEXT);

    // ---- AR loop -> codes ----
    std::vector<std::vector<int> > codes =
        run_ar_loop(rt, ie.data(), S, tth.data(), N_TEXT, tpe.data(), max_frames);
    printf("[qwen3_tts] generated %d frames\n", (int)codes.size());
    if (codes.empty()) { fprintf(stderr, "no frames generated\n"); return 1; }

    // ---- vocoder -> waveform ----
    VocoderRuntime voc;
    if (!voc.load(P)) { fprintf(stderr, "VocoderRuntime load failed\n"); return 1; }
    std::vector<float> wav = voc.decode(codes);
    printf("[qwen3_tts] waveform %d samples (%.2fs @ 24kHz)\n",
           (int)wav.size(), wav.size() / 24000.0);

    // ---- write WAV ----
    if (!save_wav(out_wav, wav.data(), (int)wav.size(), 24000))
    { fprintf(stderr, "save_wav failed\n"); return 1; }
    printf("[qwen3_tts] wrote %s\n", out_wav);
    return 0;
}
