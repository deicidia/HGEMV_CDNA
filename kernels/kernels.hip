#include "kernels.hpp"

__global__ void hgemv_block(float a, const __half* A, const __half* x, float b, __half* Y, int m, int n)
{
    (void)a; (void)A; (void)x; (void)b; (void)Y; (void)m; (void)n;
}

__global__ void hgemv_wave64(float a, const __half* A, const __half* x, float b, __half* Y, int m, int n)
{
    (void)a; (void)A; (void)x; (void)b; (void)Y; (void)m; (void)n;
}