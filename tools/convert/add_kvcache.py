# SPDX-License-Identifier: BSD-3-Clause
# Post-conversion patch: enable ncnn SDPA static-graph KV cache.
#
# pnnx converts Qwen3 attention to an ncnn `SDPA` layer but NEVER emits the
# kv_cache param (7=1) nor wires the cache-in/cache-out blobs. This script
# rewrites a .ncnn.param so that every SDPA layer:
#   - gains param 7=1 (enable kv_cache)
#   - takes two extra trailing input blobs  cache_k{i} cache_v{i}
#   - emits two extra trailing output blobs out_cache_k{i} out_cache_v{i}
# and adds one Input layer producing all cache_k*/cache_v* blobs.
#
# This matches the contract in benchmark/models/llm/qwen3_0.6b_decoder.ncnn.param
# and what benchncnn_llm.cpp resolve_cache_indexes() expects (SDPA with 3 tops).
#
# Adapted from futz12/ncnn_llm export/hunyuan_ocr_add_kvcache.py, generalized to
# auto-detect the SDPA layer count instead of hard-coding it.
#
# usage: python add_kvcache.py in.ncnn.param out.ncnn.param

import sys


def add_kvcache_to_param(input_param_path, output_param_path):
    with open(input_param_path, "r") as f:
        lines = f.readlines()

    magic = lines[0].strip()
    counts = lines[1].split()
    layer_count = int(counts[0])
    blob_count = int(counts[1])

    layer_lines = lines[2:]

    # first pass: count SDPA layers so we know how many cache blobs to declare
    num_sdpa = sum(1 for ln in layer_lines if ln.split() and ln.split()[0] == "SDPA")

    cache_in_blobs = []
    for i in range(num_sdpa):
        cache_in_blobs.append("cache_k%d" % i)
        cache_in_blobs.append("cache_v%d" % i)

    new_layer_lines = []
    sdpa_counter = 0
    for line in layer_lines:
        parts = line.split()
        if len(parts) < 2:
            new_layer_lines.append(line)
            continue

        if parts[0] == "SDPA":
            name = parts[1]
            in_count = int(parts[2])
            out_count = int(parts[3])
            idx = 4
            in_blobs = parts[idx:idx + in_count]
            idx += in_count
            out_blobs = parts[idx:idx + out_count]
            idx += out_count
            param_parts = parts[idx:]

            new_in = in_blobs + ["cache_k%d" % sdpa_counter, "cache_v%d" % sdpa_counter]
            new_out = out_blobs + ["out_cache_k%d" % sdpa_counter, "out_cache_v%d" % sdpa_counter]

            if not any(p.startswith("7=") for p in param_parts):
                param_parts.append("7=1")

            new_line = "%-24s %-24s %d %d %s %s %s\n" % (
                "SDPA", name, len(new_in), len(new_out),
                " ".join(new_in), " ".join(new_out), " ".join(param_parts),
            )
            new_layer_lines.append(new_line)
            sdpa_counter += 1
        else:
            new_layer_lines.append(line)

    # insert one Input layer producing all cache blobs, right after the last
    # top-of-file Input layer (so all blobs are produced before first use)
    insert_pos = 0
    for i, line in enumerate(new_layer_lines):
        parts = line.split()
        if len(parts) >= 2 and parts[0] == "Input":
            insert_pos = i + 1
        elif insert_pos > 0:
            break

    input_line = "%-24s %-24s %d %d %s\n" % (
        "Input", "kv_cache_input", 0, len(cache_in_blobs), " ".join(cache_in_blobs)
    )
    new_layer_lines.insert(insert_pos, input_line)

    new_layer_count = layer_count + 1
    new_blob_count = blob_count + 4 * num_sdpa  # 2 cache-in + 2 cache-out per SDPA

    with open(output_param_path, "w") as f:
        f.write(magic + "\n")
        f.write("%d %d\n" % (new_layer_count, new_blob_count))
        f.writelines(new_layer_lines)

    print("patched %s -> %s" % (input_param_path, output_param_path))
    print("  SDPA layers: %d" % num_sdpa)
    print("  layers %d -> %d, blobs %d -> %d" % (layer_count, new_layer_count, blob_count, new_blob_count))
    return num_sdpa


if __name__ == "__main__":
    if len(sys.argv) != 3:
        print("usage: python add_kvcache.py in.ncnn.param out.ncnn.param")
        sys.exit(1)
    add_kvcache_to_param(sys.argv[1], sys.argv[2])
