// SPDX-License-Identifier: Apache-2.0
// Prefill-parity test: the C++ assemble_prefill must reproduce the dumped
// golden talker_inputs_embeds.f32 (which the PyTorch reference produced).
//
// usage: test_prefill <dumps_dir> <golden_ids.txt> [spk_id] [lang_id]
//   dumps_dir must contain:
//     text_projected_table.f32   (VOCAB, 1024)
//     talker_codec_embedding.f32  (3072, 1024)
//     talker_inputs_embeds.f32    (S, 1024)   golden
#include "prefill.h"

#include <cstdio>
#include <cstdlib>
#include <cmath>
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

int main(int argc, char** argv)
{
    if (argc < 3) { fprintf(stderr, "usage: %s <dumps_dir> <golden_ids.txt> [spk_id] [lang_id]\n", argv[0]); return 1; }
    std::string D = argv[1];
    int spk_id = (argc > 3) ? atoi(argv[3]) : 3066;   // serena
    int lang_id = (argc > 4) ? atoi(argv[4]) : 2055;  // chinese

    std::vector<float> proj_table, codec_emb, golden;
    if (!load_f32(D + "/text_projected_table.f32", proj_table)) return 1;
    if (!load_f32(D + "/talker_codec_embedding.f32", codec_emb)) return 1;
    if (!load_f32(D + "/talker_inputs_embeds.f32", golden)) return 1;

    // load token ids
    FILE* f = fopen(argv[2], "r");
    if (!f) { fprintf(stderr, "open golden ids failed\n"); return 1; }
    std::vector<int> ids; int v;
    while (fscanf(f, "%d", &v) == 1) ids.push_back(v);
    fclose(f);

    qwen3_tts::PrefillIds P;
    int S = 0;
    std::vector<float> trailing;
    std::vector<float> ie = qwen3_tts::assemble_prefill(
        ids, proj_table.data(), codec_emb.data(), spk_id, lang_id, P, S, trailing);

    const int H = P.hidden;
    printf("[prefill] tokens=%zu S=%d golden_rows=%zu\n", ids.size(), S, golden.size() / H);

    int n = (int)ie.size() < (int)golden.size() ? (int)ie.size() : (int)golden.size();
    if ((int)ie.size() != (int)golden.size())
        fprintf(stderr, "WARN size mismatch: ours=%zu golden=%zu\n", ie.size(), golden.size());

    double max_abs = 0, sum_abs = 0;
    for (int i = 0; i < n; i++) {
        double d = fabs((double)ie[i] - golden[i]);
        if (d > max_abs) max_abs = d;
        sum_abs += d;
    }
    printf("[prefill] max_abs=%.3e mean_abs=%.3e\n", max_abs, sum_abs / n);
    bool ok = (max_abs < 1e-3) && (ie.size() == golden.size());
    printf("[prefill] MATCH: %s\n", ok ? "YES" : "NO");
    return ok ? 0 : 2;
}
