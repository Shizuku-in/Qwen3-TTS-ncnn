# SPDX-License-Identifier: BSD-3-Clause
# Post-conversion patch: make the Talker decoder graph seqlen-dynamic.
#
# The clean STATIC pnnx export bakes the trace-time sequence length into the
# q/k/v/o Reshape ops (e.g. `2=8`), locking the graph to that seqlen. The AR
# loop needs variable prefill length + seq=1 decode. This script rewrites the
# baked seqlen dim to -1 (ncnn's "infer" marker), producing exactly the pattern
# the shipped benchmark/models/llm/qwen3_0.6b_decoder.ncnn.param uses.
#
# Resolved dim mapping (S = trace seqlen, e.g. 8), verified against the shipped
# decoder's attention block:
#   q reshape:      0=128 1=16 2=S   -> 0=128 1=16 2=-1   (heads=16 const, dim2=seq)
#   post-GQA k/v:   0=128 1=S  2=16  -> 0=128 1=-1 2=16   (dim1=seq, heads=16 const)
#   k/v reshape:    0=128 1=8  2=S   -> 0=128 1=8  2=-1   (kv_heads=8 const, dim2=seq)
#   o_proj flatten: 0=2048 1=S       -> 0=2048 1=-1       (dim1=seq, 2048=16*128)
#
# Only the seq dim is touched; head-count / head-dim constants are preserved.
#
# usage: python dynamize_seqlen.py in.ncnn.param out.ncnn.param SEQLEN

import sys


def dynamize(in_path, out_path, S):
    S = int(S)
    # exact (old -> new) reshape param-tail rewrites. Keyed on the full tail so
    # we never touch a reshape whose matching dim is a genuine constant.
    rewrites = {
        f"0=128 1=16 2={S}": "0=128 1=16 2=-1",   # q: [head_dim, n_heads, seq]
        f"0=128 1={S} 2=16": "0=128 1=-1 2=16",   # post-GQA k/v: [head_dim, seq, n_heads]
        f"0=128 1=8 2={S}": "0=128 1=8 2=-1",     # k/v: [head_dim, n_kv_heads, seq]
        f"0=2048 1={S}": "0=2048 1=-1",           # o_proj flatten: [hidden, seq]
    }

    with open(in_path) as f:
        lines = f.readlines()

    counts = {k: 0 for k in rewrites}
    ambiguous = 0
    out_lines = []
    for line in lines:
        parts = line.split()
        if parts and parts[0] == "Reshape":
            # split into "head" (up to blob names) and "param tail"
            # format: Reshape name in out b0 b1 <params...>
            head = parts[:6]  # Reshape name incount outcount in0 out0
            tail = " ".join(parts[6:])
            if tail in rewrites:
                # guard against the genuinely ambiguous 0=128 1=8 2=8 (both dims S=8):
                # only the k/v pattern has kv_heads==8; if S==8 the q-pattern
                # 0=128 1=16 2=8 and post-GQA 0=128 1=8 2=16 are unambiguous, but
                # 0=128 1=8 2=8 could be k/v (dim2=seq) — which is what we want.
                new_tail = rewrites[tail]
                line = "%-24s %-24s %s %s %s %s %s\n" % (
                    head[0], head[1], head[2], head[3], head[4], head[5], new_tail
                )
                counts[tail] += 1
        out_lines.append(line)

    with open(out_path, "w") as f:
        f.writelines(out_lines)

    print(f"dynamize {in_path} -> {out_path} (S={S})")
    total = 0
    for k, v in counts.items():
        print(f"  '{k}' -> '{rewrites[k]}' : {v} sites")
        total += v
    print(f"  total rewrites: {total}")
    if total == 0:
        print("  WARNING: no rewrites applied — check SEQLEN matches the trace seqlen")
    return total


if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("usage: python dynamize_seqlen.py in.ncnn.param out.ncnn.param SEQLEN")
        sys.exit(1)
    dynamize(sys.argv[1], sys.argv[2], sys.argv[3])
