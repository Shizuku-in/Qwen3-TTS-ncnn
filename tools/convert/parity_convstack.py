# SPDX-License-Identifier: Apache-2.0
# Parity: converted conv tail (pre_conv skipped; full post-transformer conv stack) vs PyTorch.
import argparse, os, numpy as np, torch, torch.nn as nn

class ConvTail(nn.Module):
    def __init__(self, dec):
        super().__init__()
        self.upsample = dec.upsample
        self.decoder = dec.decoder
    def forward(self, x):
        for blocks in self.upsample:
            for b in blocks:
                x = b(x)
        for b in self.decoder:
            x = b(x)
        return x.clamp(min=-1, max=1)

def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", required=True)
    ap.add_argument("--outdir", default="tools/qwen3_tts/pnnx_out")
    ap.add_argument("--frames", type=int, default=8)
    args = ap.parse_args()
    from qwen_tts import Qwen3TTSModel
    print("[pc] loading ...", flush=True)
    w = Qwen3TTSModel.from_pretrained(args.model, device_map="cpu", dtype=torch.float32, attn_implementation="sdpa")
    dec = w.model.speech_tokenizer.model.decoder
    mod = ConvTail(dec).eval()
    T = args.frames
    torch.manual_seed(0)
    x = torch.randn(1, 1024, T, dtype=torch.float32)
    with torch.no_grad():
        ref = mod(x).squeeze(0).squeeze(0).numpy()  # (samples,)
    print(f"[pc] torch out {ref.shape}", flush=True)
    import ncnn
    with ncnn.Net() as net:
        net.load_param(os.path.join(args.outdir, "dec_conv_tail.ncnn.param"))
        net.load_model(os.path.join(args.outdir, "dec_conv_tail.ncnn.bin"))
        with net.create_extractor() as ex:
            ex.input("in0", ncnn.Mat(x.squeeze(0).numpy()).clone())
            _, out0 = ex.extract("out0")
            ncnn_out = np.array(out0)
    ncnn_out = ncnn_out.reshape(-1)
    print(f"[pc] shapes ref={ref.shape} ncnn={ncnn_out.shape}", flush=True)
    n = min(ref.shape[0], ncnn_out.shape[0])
    d = np.abs(ref[:n] - ncnn_out[:n])
    cs = float(np.dot(ref[:n], ncnn_out[:n]) / (np.linalg.norm(ref[:n])*np.linalg.norm(ncnn_out[:n]) + 1e-9))
    print(f"[pc] len ref={ref.shape[0]} ncnn={ncnn_out.shape[0]} max_abs={d.max():.3e} mean={d.mean():.3e} cos={cs:.6f}", flush=True)
    print(f"[pc] VERDICT: {'PASS' if (cs>0.9999 and ref.shape[0]==ncnn_out.shape[0]) else 'FAIL'}", flush=True)

if __name__ == "__main__":
    main()
