# coding=utf-8
# Generate a deterministic synthetic 24kHz reference wav for parity testing.
# Not meant to sound like speech — only to give the speaker encoder + voice-clone
# path a fixed, reproducible input so torch vs ncnn can be compared bit-for-bit.
import numpy as np
import soundfile as sf
import os

SR = 24000
DUR = 3.0  # seconds
OUT = os.path.join(os.path.dirname(__file__), "..", "dumps", "ref_input.wav")
OUT = os.path.abspath(OUT)

rng = np.random.RandomState(1234)
t = np.linspace(0, DUR, int(SR * DUR), endpoint=False)

# speech-like: a few formant-ish sinusoids with slow AM, + a little noise
sig = np.zeros_like(t)
for f0, amp in [(120, 0.5), (240, 0.25), (480, 0.15), (900, 0.08)]:
    am = 0.6 + 0.4 * np.sin(2 * np.pi * 3.0 * t + f0)  # slow amplitude modulation
    sig += amp * am * np.sin(2 * np.pi * f0 * t)
sig += 0.01 * rng.randn(len(t))
sig = sig / np.max(np.abs(sig)) * 0.9

sf.write(OUT, sig.astype(np.float32), SR)
print("wrote", OUT, "samples", len(sig), "sr", SR)
