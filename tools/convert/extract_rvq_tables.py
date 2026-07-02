# coding=utf-8
# SPDX-License-Identifier: Apache-2.0
#
# Extract RVQ dequant gather tables for the 12Hz speech decoder, folding the
# per-group output_proj (256->512, Conv1d k=1, bias=False) into each codebook
# table so the C++ runtime RVQ dequant is pure gather + sum (zero conv).
#
# Decode math (from modeling_qwen3_tts_tokenizer_v2.py):
#   EuclideanCodebook.decode: emb = embedding_sum / clamp(cluster_usage, eps)   # (bins, 256)
#                             q   = F.embedding(codes, emb)                       # (..., 256)
#   ResidualVectorQuantization.decode: sum over the group's codebooks of the lookup
#   ResidualVectorQuantizer.decode: output_proj( sum )   # Conv1d 256->512 k=1
#   SplitResidualVectorQuantizer.decode: rvq_first(codes[:1]) + rvq_rest(codes[1:])
#
# Since output_proj is linear, output_proj(sum_q lookup_q) == sum_q output_proj(lookup_q).
# So we fold: table[q][c] = output_proj_group @ (embedding_sum_q[c] / clamp(cluster_usage_q[c]))
# giving a (bins, 512) table per codebook. Runtime dequant for a frame:
#   hidden[:, t] = sem_table[code0_t] + sum_{q=1..15} acoustic_table[q-1][code_q_t]
#
# Output: a single binary file `rvq_tables.bin` = 16 contiguous (2048, 512) f32
# tables (index 0 = semantic/rvq_first, 1..15 = acoustic/rvq_rest layers 0..14),
# plus a small `rvq_tables.meta.txt` describing shapes.

import argparse
import os
import numpy as np
import torch


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", default="tools/qwen3_tts/pnnx_out")
    args = ap.parse_args()
    os.makedirs(args.outdir, exist_ok=True)

    from qwen_tts import Qwen3TTSModel

    def log(m):
        print(f"[rvq] {m}", flush=True)

    log(f"loading {args.model} (fp32, cpu) ...")
    wrapper = Qwen3TTSModel.from_pretrained(
        args.model, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa"
    )
    quant = wrapper.model.speech_tokenizer.model.decoder.quantizer
    eps = quant.rvq_first.vq.layers[0]._codebook.epsilon
    log(f"epsilon={eps}")

    def fold_group(rvq):
        # output_proj: Conv1d(256->512, k=1, bias=False), weight (512,256,1)
        W = rvq.output_proj.weight.detach().squeeze(-1)  # (512, 256)
        tables = []
        for layer in rvq.vq.layers:
            cb = layer._codebook
            emb = cb.embedding_sum.detach() / cb.cluster_usage.detach().clamp(min=cb.epsilon)[:, None]  # (bins,256)
            # project_out is Identity, so proj is only output_proj: (bins,256) @ (256,512) = (bins,512)
            tbl = emb @ W.T  # (bins, 512)
            tables.append(tbl.contiguous())
        return tables

    sem_tables = fold_group(quant.rvq_first)      # 1 table
    aco_tables = fold_group(quant.rvq_rest)        # 15 tables
    all_tables = sem_tables + aco_tables           # 16 total, index 0=semantic
    log(f"tables: {len(all_tables)} (sem={len(sem_tables)} aco={len(aco_tables)})")
    for i, t in enumerate(all_tables):
        assert t.shape[1] == 512, t.shape
    bins = all_tables[0].shape[0]

    # write binary: 16 x (bins, 512) f32 row-major
    binpath = os.path.join(args.outdir, "rvq_tables.bin")
    with open(binpath, "wb") as f:
        for t in all_tables:
            f.write(t.numpy().astype(np.float32).tobytes())
    metapath = os.path.join(args.outdir, "rvq_tables.meta.txt")
    with open(metapath, "w") as f:
        f.write(f"num_tables 16\nbins {bins}\ndim 512\n")
        f.write("order table0=semantic(rvq_first), table1..15=acoustic(rvq_rest layers 0..14)\n")
    log(f"wrote {binpath} ({os.path.getsize(binpath)} bytes) + {metapath}")

    # ---- parity: fold tables vs real quantizer.decode ----
    log("parity vs real quantizer.decode ...")
    T = 12
    torch.manual_seed(0)
    codes = torch.randint(0, bins, (1, 16, T), dtype=torch.long)  # (B,16,T)
    with torch.no_grad():
        ref = quant.decode(codes)  # (B,512,T)
    ref = ref[0].transpose(0, 1).numpy()  # (T,512)

    # our fold: per frame sum table lookups
    tbl = np.stack([t.numpy() for t in all_tables], axis=0)  # (16,bins,512)
    cds = codes[0].numpy()  # (16,T)
    ours = np.zeros((T, 512), dtype=np.float32)
    for t in range(T):
        acc = tbl[0, cds[0, t]].copy()
        for q in range(1, 16):
            acc += tbl[q, cds[q, t]]
        ours[t] = acc

    diff = np.abs(ref - ours)
    cos = float((ref * ours).sum() / (np.linalg.norm(ref) * np.linalg.norm(ours) + 1e-12))
    log(f"max_abs_diff={diff.max():.3e} mean={diff.mean():.3e} cos_sim={cos:.6f}")
    log("VERDICT: " + ("PASS" if diff.max() < 1e-4 else "FAIL"))


if __name__ == "__main__":
    main()
