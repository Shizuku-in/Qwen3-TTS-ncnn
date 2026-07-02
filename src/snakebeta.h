// SPDX-License-Identifier: Apache-2.0
// SnakeBeta activation as an ncnn custom layer.
//
// This is the ONE op in the speech decoder that pnnx cannot lower cleanly
// (its per-channel alpha/beta broadcast gets mangled by the expression folder),
// so we keep it as a moduleop during conversion and implement it here.
//
//   SnakeBeta(x) = x + (1 / (exp(beta) + eps)) * sin(x * exp(alpha))^2
//
// alpha and beta are per-channel (length C) learnable parameters. Input layout
// is ncnn (w, h, c) where c is the feature/channel dim — i.e. one alpha/beta
// per channel, applied across the whole (w,h) plane of that channel.
//
// pnnx emits this op (via moduleop) with two array params carrying the weight
// shapes and writes the raw alpha/beta floats into the .bin, consumed here in
// load_model().
#pragma once

#include "layer.h"
#include "mat.h"

#include <math.h>

namespace qwen3_tts {

class SnakeBeta : public ncnn::Layer
{
public:
    SnakeBeta()
    {
        one_blob_only = true;
        support_inplace = true;
    }

    virtual int load_param(const ncnn::ParamDict& pd)
    {
        // pnnx writes the attribute shapes as array params 10 (alpha) and 11
        // (beta), each an array whose last element is the channel count.
        ncnn::Mat alpha_shape = pd.get(10, ncnn::Mat());
        ncnn::Mat beta_shape = pd.get(11, ncnn::Mat());

        num_features = 0;
        if (!alpha_shape.empty())
        {
            const int* p = alpha_shape;
            num_features = p[alpha_shape.w - 1];
        }
        else if (!beta_shape.empty())
        {
            const int* p = beta_shape;
            num_features = p[beta_shape.w - 1];
        }
        return 0;
    }

    virtual int load_model(const ncnn::ModelBin& mb)
    {
        if (num_features <= 0)
            return -1;

        // raw float arrays, in the order pnnx serialized them (alpha then beta)
        alpha = mb.load(num_features, 1);
        if (alpha.empty())
            return -100;
        beta = mb.load(num_features, 1);
        if (beta.empty())
            return -100;

        // precompute exp(alpha) and 1/(exp(beta)+eps) per channel
        exp_alpha.create(num_features);
        inv_beta.create(num_features);
        if (exp_alpha.empty() || inv_beta.empty())
            return -100;

        const float eps = 1e-9f;
        const float* ap = alpha;
        const float* bp = beta;
        float* eap = exp_alpha;
        float* ibp = inv_beta;
        for (int i = 0; i < num_features; i++)
        {
            eap[i] = expf(ap[i]);
            ibp[i] = 1.f / (expf(bp[i]) + eps);
        }
        return 0;
    }

    virtual int forward_inplace(ncnn::Mat& bottom_top_blob, const ncnn::Option& opt) const
    {
        const int w = bottom_top_blob.w;
        const int h = bottom_top_blob.h;
        const int d = bottom_top_blob.d;
        const int channels = bottom_top_blob.c;
        const int size = w * h * d;

        const float* eap = exp_alpha;
        const float* ibp = inv_beta;

        #pragma omp parallel for num_threads(opt.num_threads)
        for (int q = 0; q < channels; q++)
        {
            float* ptr = bottom_top_blob.channel(q);
            const float a = eap[q];
            const float ib = ibp[q];
            for (int i = 0; i < size; i++)
            {
                const float s = sinf(ptr[i] * a);
                ptr[i] = ptr[i] + ib * s * s;
            }
        }
        return 0;
    }

public:
    int num_features;
    ncnn::Mat alpha;
    ncnn::Mat beta;
    ncnn::Mat exp_alpha;
    ncnn::Mat inv_beta;
};

DEFINE_LAYER_CREATOR(SnakeBeta)

} // namespace qwen3_tts
