#include <iostream>
#include <algorithm>
#include <iomanip>
#include <random>
#include <string>
#include <vector>

#include <thrust/copy.h>
#include <thrust/device_vector.h>

#include "hip_utils.hpp"
#include "variants.hpp"

namespace
{
constexpr int    kWarmup   = 3;         // absorbs code-object load and clock ramp-up
constexpr double kBudgetMs = 300.0;     // target total time for the benchmark, used to determine the number of iterations
constexpr int    kMinIters = 3;
constexpr int    kMaxIters = 200;
constexpr double kPeakBps  = 613e9;     // BabelStream 5.0, Copy, float, 403 MB

struct Result
{
    double ms;
    double bw;
};

auto benchmark(const Variant& v, int m, int n,
               const thrust::device_vector<float>& d_A,
               const thrust::device_vector<float>& d_x,
               thrust::device_vector<float>& d_Y) -> Result
{
    const int grid = grid_for(v, m);

    auto launch = [&] {
        v.kernel<<<grid, kBlockSize>>>(
            1.0f,
            thrust::raw_pointer_cast(d_A.data()),
            thrust::raw_pointer_cast(d_x.data()),
            0.0f,
            thrust::raw_pointer_cast(d_Y.data()),
            m, n);
    };

    hipEvent_t start, stop;
    HIP_CHECK(hipEventCreate(&start));
    HIP_CHECK(hipEventCreate(&stop));

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < kWarmup; ++i) launch();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float warm_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&warm_ms, start, stop));

    const double per_launch = warm_ms / kWarmup;
    const int iters = std::clamp(static_cast<int>(kBudgetMs / per_launch), kMinIters, kMaxIters);

    HIP_CHECK(hipEventRecord(start));
    for (int i = 0; i < iters; ++i) launch();
    HIP_CHECK(hipEventRecord(stop));
    HIP_CHECK(hipEventSynchronize(stop));

    float total_ms = 0.0f;
    HIP_CHECK(hipEventElapsedTime(&total_ms, start, stop));
    HIP_CHECK(hipEventDestroy(start));
    HIP_CHECK(hipEventDestroy(stop));

    const double useful = static_cast<double>(m) * n + n + 2.0 * m;
    const double secs   = (total_ms / 1e3) / iters;
    return {secs * 1e3, useful * sizeof(float) / secs};
}
}

auto main() -> int 
{
    constexpr int  kMinDim = 1 << 8;    // 256
    constexpr int  kMaxDim = 1 << 22;   // 4194304
    constexpr long kElems  = 1L << 30;  // m * n, constant

    std::cerr << "Sweep Benchmarking\n";
    
    std::vector<float> A(kElems);
    std::vector<float> x(kMaxDim);
    std::vector<float> Y(kMaxDim);

    std::mt19937 gen{42};
    std::uniform_real_distribution<float> dis{-1.0f, 1.0f};
    std::generate(A.begin(), A.end(), [&] { return dis(gen); });
    std::generate(x.begin(), x.end(), [&] { return dis(gen); });
    std::generate(Y.begin(), Y.end(), [&] { return dis(gen); });

    thrust::device_vector<float> d_A = A;
    thrust::device_vector<float> d_x = x;
    thrust::device_vector<float> d_Y = Y;

    const auto& selected = variants();

    std::cout << "variant,m,n,grid,ms,gbs,pct_of_peak\n";

    for(int i = 0; i <= 14; ++i)
    {
        const int row = kMinDim << i;
        const int col = kMaxDim >> i;

        for(const auto& v : selected)
        {
            const Result r = benchmark(v, row, col, d_A, d_x, d_Y);

            std::cout << v.name << ',' << row << ',' << col << ','
                      << grid_for(v, row) << ','
                      << std::fixed << std::setprecision(4) << r.ms << ','
                      << std::setprecision(1) << r.bw / 1e9 << ','
                      << std::setprecision(1) << 100.0 * r.bw / kPeakBps << '\n';
            std::cout.flush();
        }
    }

    return EXIT_SUCCESS;
}