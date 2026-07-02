// SPDX-License-Identifier: Apache-2.0
// Qwen3-TTS ncnn synthesizer (self-contained: raw text -> 24 kHz WAV).
//
// Full pure-C++/ncnn pipeline for the 12Hz CustomVoice checkpoint:
//   text --tokenizer--> ids --prefill--> inputs_embeds
//        --AR loop (Talker + Code Predictor)--> codes [T,16]
//        --vocoder (RVQ -> pre_conv -> transformer -> conv tail)--> waveform
//
// Every stage is validated against the PyTorch reference (see the tests/ parity
// harnesses and memory qwen3-tts-cpp-integration).
//
// usage: qwen3_tts <pnnx_dir> <assets_dir> <out.wav> [text] [speaker] [language] [max_frames]
//   pnnx_dir:   converted ncnn nets (talker/cp/decoder + rvq_tables.bin)
//   assets_dir: vocab.txt, merges.txt, text_projected_table.f32,
//               talker_codec_embedding.f32, and the AR-loop weight tables
#include "pipeline.h"
#include "prefill.h"
#include "wav.h"
#include "tokenizer/bpe_tokenizer.h"

#include <stdio.h>
#include <string>
#include <vector>
#include <map>

// preset speaker -> codec id (from config.json talker_config.spk_id)
static int speaker_id(const std::string& s)
{
    static const std::map<std::string, int> m = {
        {"serena", 3066}, {"vivian", 3065}, {"uncle_fu", 3010}, {"ryan", 3061},
        {"aiden", 2861}, {"ono_anna", 2873}, {"sohee", 2864}, {"eric", 2875}, {"dylan", 2878}};
    auto it = m.find(s);
    return it == m.end() ? -1 : it->second;
}

// language -> codec id (from config.json talker_config.codec_language_id)
static int language_id(const std::string& s)
{
    static const std::map<std::string, int> m = {
        {"chinese", 2055}, {"english", 2050}, {"german", 2053}, {"italian", 2070},
        {"portuguese", 2071}, {"spanish", 2054}, {"japanese", 2058}, {"korean", 2064},
        {"french", 2061}, {"russian", 2069}, {"beijing_dialect", 2074}, {"sichuan_dialect", 2062}};
    auto it = m.find(s);
    return it == m.end() ? -1 : it->second;
}

int main(int argc, char** argv)
{
    if (argc < 4)
    {
        fprintf(stderr, "usage: %s <pnnx_dir> <assets_dir> <out.wav> [text] [speaker] [language] [max_frames]\n", argv[0]);
        return 1;
    }
    const std::string P = argv[1];
    const std::string A = argv[2];
    const char* out_wav = argv[3];
    const std::string text = (argc > 4) ? argv[4] : "\xe8\xbf\x99\xe6\x98\xaf\xe4\xb8\x80\xe4\xb8\xaa\xe6\xb5\x8b\xe8\xaf\x95\xe3\x80\x82";
    const std::string speaker = (argc > 5) ? argv[5] : "serena";
    const std::string language = (argc > 6) ? argv[6] : "chinese";
    const int max_frames = (argc > 7) ? atoi(argv[7]) : 256;

    using namespace qwen3_tts;
    typedef TalkerDims TD;
    const int H = TD::HIDDEN;

    const int spk = speaker_id(speaker);
    const int lang = language_id(language);
    if (spk < 0) { fprintf(stderr, "unknown speaker '%s'\n", speaker.c_str()); return 1; }
    if (lang < 0) { fprintf(stderr, "unknown language '%s'\n", language.c_str()); return 1; }

    // ---- tokenize ChatML-wrapped text ----
    SpecialTokensConfig spec;
    spec.bos_token = "<|endoftext|>";
    spec.eos_token = "<|im_end|>";
    BpeTokenizer tok = BpeTokenizer::LoadFromFiles(
        (A + "/vocab.txt").c_str(), (A + "/merges.txt").c_str(), spec, true, true, true);
    const std::string chatml = "<|im_start|>assistant\n" + text + "<|im_end|>\n<|im_start|>assistant\n";
    std::vector<int> ids = tok.encode(chatml, false, false, false, false);
    printf("[qwen3_tts] %zu tokens, speaker=%s(%d) lang=%s(%d)\n",
           ids.size(), speaker.c_str(), spk, language.c_str(), lang);

    // ---- load tables + assemble prefill ----
    std::vector<float> proj_table, codec_emb;
    if (!load_f32(A + "/text_projected_table.f32", proj_table)) { fprintf(stderr, "proj_table load failed\n"); return 1; }
    if (!load_f32(A + "/talker_codec_embedding.f32", codec_emb)) { fprintf(stderr, "codec_emb load failed\n"); return 1; }

    PrefillIds PI;
    int S = 0;
    std::vector<float> trailing;
    std::vector<float> ie = assemble_prefill(ids, proj_table.data(), codec_emb.data(),
                                             spk, lang, PI, S, trailing);
    printf("[qwen3_tts] prefill S=%d\n", S);

    // ---- AR loop -> codes ----
    TalkerRuntime rt;
    if (!rt.load(P, A)) { fprintf(stderr, "TalkerRuntime load failed\n"); return 1; }
    std::vector<std::vector<int> > codes =
        run_ar_loop(rt, ie.data(), S, trailing.data(), 1, trailing.data(), max_frames);
    printf("[qwen3_tts] generated %d frames\n", (int)codes.size());
    if (codes.empty()) { fprintf(stderr, "no frames generated\n"); return 1; }

    // ---- vocoder -> waveform ----
    VocoderRuntime voc;
    if (!voc.load(P)) { fprintf(stderr, "VocoderRuntime load failed\n"); return 1; }
    std::vector<float> wav = voc.decode(codes);
    printf("[qwen3_tts] waveform %d samples (%.2fs @ 24kHz)\n",
           (int)wav.size(), wav.size() / 24000.0);

    if (!save_wav(out_wav, wav.data(), (int)wav.size(), 24000))
    { fprintf(stderr, "save_wav failed\n"); return 1; }
    printf("[qwen3_tts] wrote %s\n", out_wav);
    return 0;
}
