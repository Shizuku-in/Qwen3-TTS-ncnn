// SPDX-License-Identifier: Apache-2.0
// Token-parity test: the vendored BPE tokenizer must reproduce the exact token
// IDs the Python Qwen2Tokenizer produces for the ChatML-wrapped text.
//
// usage: test_tokenizer <vocab.txt> <merges.txt> <golden_ids.txt>
//   golden_ids.txt: whitespace-separated int token ids for the FULL chatml text
//   <|im_start|>assistant\n{text}<|im_end|>\n<|im_start|>assistant\n
#include "tokenizer/bpe_tokenizer.h"
#include <cstdio>
#include <cstdlib>
#include <string>
#include <vector>

int main(int argc, char** argv)
{
    if (argc < 4) { fprintf(stderr, "usage: %s <vocab.txt> <merges.txt> <golden_ids.txt>\n", argv[0]); return 1; }
    SpecialTokensConfig spec;
    spec.bos_token = "<|endoftext|>";
    spec.eos_token = "<|im_end|>";
    BpeTokenizer tok = BpeTokenizer::LoadFromFiles(argv[1], argv[2], spec, true, true, true);

    // register the ChatML / tts special tokens so they encode as single ids
    tok.SetAdditionalSpecialTokens({"<|im_start|>", "<|im_end|>"}, true);

    // full ChatML text (matches _build_assistant_text in the reference)
    const std::string text =
        "<|im_start|>assistant\n"
        "\xe8\xbf\x99\xe6\x98\xaf\xe4\xb8\x80\xe4\xb8\xaa\xe7\x94\xa8\xe4\xba\x8e\xe9\xaa\x8c\xe8\xaf\x81 ncnn "
        "\xe7\xa7\xbb\xe6\xa4\x8d\xe6\x95\xb0\xe5\x80\xbc\xe4\xb8\x80\xe8\x87\xb4\xe6\x80\xa7\xe7\x9a\x84\xe6\xb5\x8b\xe8\xaf\x95\xe5\x8f\xa5\xe5\xad\x90\xe3\x80\x82"
        "<|im_end|>\n<|im_start|>assistant\n";

    std::vector<int> ids = tok.encode(text, false, false, false, false);

    // load golden
    FILE* f = fopen(argv[3], "r");
    if (!f) { fprintf(stderr, "open golden failed\n"); return 1; }
    std::vector<int> gold; int v;
    while (fscanf(f, "%d", &v) == 1) gold.push_back(v);
    fclose(f);

    printf("[tok] ours=%zu golden=%zu\n", ids.size(), gold.size());
    printf("[tok] ours:  ");
    for (int id : ids) printf("%d ", id);
    printf("\n[tok] gold:  ");
    for (int id : gold) printf("%d ", id);
    printf("\n");

    bool match = ids.size() == gold.size();
    for (size_t i = 0; match && i < ids.size(); i++) if (ids[i] != gold[i]) match = false;
    printf("[tok] MATCH: %s\n", match ? "YES" : "NO");
    return match ? 0 : 2;
}
