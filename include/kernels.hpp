#pragma once

#include <hip/hip_runtime.h>
#include <hip/hip_fp16.h>


__global__ void hgemv_block(float a, const __half* A, const __half* x, float b, __half* Y, int m, int n);

__global__ void hgemv_wave64(float a, const __half* A, const __half* x, float b, __half* Y, int m, int n);

