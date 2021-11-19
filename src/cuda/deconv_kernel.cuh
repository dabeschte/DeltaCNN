#pragma once
#include <stdint.h>
#include <cuda_fp16.h>
#include "common.cuh"
#include <vector>


template<typename scalar_t>
void delta_deconv(scalar_t *input, scalar_t *output, scalar_t *filter, scalar_t *bias, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config);

void delta_deconv_hp(half *input, half *output, half *filter, half *bias, uint32_t *mask, uint32_t *out_mask, Dimensions dim, ConvConfig config);

void init_d_metrics_deconv_kernels();
