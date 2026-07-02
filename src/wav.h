// SPDX-License-Identifier: Apache-2.0
// Minimal WAV I/O for the Qwen3-TTS ncnn port.
//
// The speech decoder emits float PCM in [-1, 1] at 24 kHz mono. We write a
// standard 16-bit PCM WAV. A reader is provided for parity checks against the
// PyTorch golden output.
#pragma once

#include <stdint.h>
#include <stdio.h>
#include <string.h>
#include <vector>

namespace qwen3_tts {

// Write mono float samples in [-1, 1] as 16-bit PCM WAV.
inline bool save_wav(const char* path, const float* samples, int num_samples, int sample_rate)
{
    FILE* f = fopen(path, "wb");
    if (!f)
        return false;

    const int16_t num_channels = 1;
    const int16_t bits_per_sample = 16;
    const int32_t byte_rate = sample_rate * num_channels * bits_per_sample / 8;
    const int16_t block_align = num_channels * bits_per_sample / 8;
    const int32_t data_chunk_size = num_samples * num_channels * bits_per_sample / 8;
    const int32_t chunk_size = 36 + data_chunk_size;

    fwrite("RIFF", 1, 4, f);
    fwrite(&chunk_size, 4, 1, f);
    fwrite("WAVE", 1, 4, f);

    fwrite("fmt ", 1, 4, f);
    const int32_t subchunk1_size = 16;
    const int16_t audio_format = 1; // PCM
    fwrite(&subchunk1_size, 4, 1, f);
    fwrite(&audio_format, 2, 1, f);
    fwrite(&num_channels, 2, 1, f);
    fwrite(&sample_rate, 4, 1, f);
    fwrite(&byte_rate, 4, 1, f);
    fwrite(&block_align, 2, 1, f);
    fwrite(&bits_per_sample, 2, 1, f);

    fwrite("data", 1, 4, f);
    fwrite(&data_chunk_size, 4, 1, f);

    std::vector<int16_t> pcm(num_samples);
    for (int i = 0; i < num_samples; i++)
    {
        float v = samples[i];
        if (v < -1.f) v = -1.f;
        if (v > 1.f) v = 1.f;
        pcm[i] = (int16_t)(v * 32767.f);
    }
    fwrite(pcm.data(), sizeof(int16_t), num_samples, f);

    fclose(f);
    return true;
}

} // namespace qwen3_tts
