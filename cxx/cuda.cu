/**
 * @file  cuda.cu
 * @brief The CUDA backend: cuFFT, batches of windows of one or more frames
 *
 * A group of frames is copied to the GPU with its summed-area tables; its windows (pairs of a frame
 * and a reference window) are correlated in batches: the search windows extracted, transformed,
 * multiplied by the conjugate spectra of the reference windows (computed once), transformed back,
 * normalized in place, and the peak of each surface found by a block.
 */

#include <cub/block/block_scan.cuh>
#include <cuda_runtime.h>
#include <cufft.h>

#include <algorithm>
#include <map>
#include <stdexcept>
#include <string>
#include <utility>
#include <vector>

#include "framecor.h"
#include "surface.h"

namespace framecor {
namespace {

void check(cudaError_t e, const char *call)
{
    if (e != cudaSuccess)
        throw std::runtime_error(std::string("framecor: ") + call + ": " + cudaGetErrorString(e));
}

void check(cufftResult r, const char *call)
{
    if (r != CUFFT_SUCCESS)
        throw std::runtime_error(std::string("framecor: ") + call + ": cuFFT error " + std::to_string(int(r)));
}

#define CHECK(call) check((call), #call)

constexpr int THREADS = 256;

size_t divUp(size_t a, size_t b) { return (a + b - 1) / b; }

/// the parameters of the kernels
struct Grid {
    int h, w;                   // the frame size
    int W, R, L, Lh, M;         // the window, the half search range, the search window, the width
                                // of its spectrum, the surface
    int nx, nwin;               // the windows across, all the windows
    int stat;                   // the size of the SNR window
    size_t satSize;             // the size of a summed-area table, (h + 1) x (w + 1)
    const int *down, *across;   // the first pixels of the windows
    const float *norm;          // the norms of the zero-mean reference windows
    const int *gross;           // the gross offsets of the pairs of the group, or nullptr
};

/// the search window of the pair q of the group (frame q / nwin, window q % nwin): its first pixel
/// and its gross offset; whether it is valid
__device__ inline bool pairWindow(const Grid &g, int q, int &sy, int &sx, int &gy, int &gx)
{
    const int k = q % g.nwin;
    gy = g.gross ? g.gross[2 * q] : 0;
    gx = g.gross ? g.gross[2 * q + 1] : 0;
    return g.norm[k] > 0
        && searchWindow(g.down[k / g.nx], g.across[k % g.nx], gy, gx, g.R, g.L, g.h, g.w, sy, sx);
}

/// the row prefix sums of the pixels and of their squares, in the rows 1..h of the tables; one
/// block per row of a frame (blockIdx.y)
__global__ void satRows(const uint8_t *frames, int h, int w, size_t satSize, int64_t *sat, int64_t *sat2)
{
    using Scan = cub::BlockScan<long long, THREADS>;
    __shared__ typename Scan::TempStorage tmp;
    const int y = blockIdx.x, f = blockIdx.y;
    const uint8_t *row = frames + (size_t(f) * h + y) * w;
    int64_t *s = sat + f * satSize + size_t(y + 1) * (w + 1);
    int64_t *s2 = sat2 + f * satSize + size_t(y + 1) * (w + 1);
    const int chunk = (w + THREADS - 1) / THREADS;
    const int x0 = threadIdx.x * chunk, x1 = min(x0 + chunk, w);
    long long a = 0, b = 0;
    for (int x = x0; x < x1; x++) {
        a += row[x];
        b += row[x] * row[x];
    }
    Scan(tmp).ExclusiveSum(a, a);
    __syncthreads();
    Scan(tmp).ExclusiveSum(b, b);
    for (int x = x0; x < x1; x++) {
        a += row[x];
        b += row[x] * row[x];
        s[x + 1] = a;
        s2[x + 1] = b;
    }
    if (threadIdx.x == 0) s[0] = s2[0] = 0;
}

/// the column sums of the row prefix sums: the summed-area tables; one thread per column of a frame
__global__ void satColumns(int h, int w, size_t satSize, int64_t *sat, int64_t *sat2)
{
    const int x = blockIdx.x * blockDim.x + threadIdx.x + 1, f = blockIdx.y;
    if (x > w) return;
    int64_t *s = sat + f * satSize, *s2 = sat2 + f * satSize;
    const size_t stride = w + 1;
    for (int y = 2; y <= h; y++) {
        s[y * stride + x] += s[(y - 1) * stride + x];
        s2[y * stride + x] += s2[(y - 1) * stride + x];
    }
}

/// the search windows of the pairs q0..q0 + count - 1, their means subtracted (zeros if not valid)
__global__ void extract(Grid g, const uint8_t *frames, const int64_t *sat, int q0, int count, float *buf)
{
    const size_t LL = size_t(g.L) * g.L;
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count * LL) return;
    const int p = int(idx / LL), k = int(idx % LL), i = k / g.L, j = k % g.L;
    const int q = q0 + p, f = q / g.nwin;
    int sy, sx, gy, gx;
    if (!pairWindow(g, q, sy, sx, gy, gx)) {
        buf[idx] = 0.0f;
        return;
    }
    const float mean = float(boxSum(sat + f * g.satSize, g.w + 1, sy, sx, g.L)) / float(LL);
    buf[idx] = frames[(size_t(f) * g.h + sy + i) * g.w + sx + j] - mean;
}

/// the spectra of the search windows times the conjugate spectra of their reference windows
__global__ void multiplyConjugate(cufftComplex *spec, const cufftComplex *ref, int q0, int count, int nwin, int S)
{
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= size_t(count) * S) return;
    const int p = int(idx / S), k = int(idx % S);
    const cufftComplex a = spec[idx], b = ref[size_t((q0 + p) % nwin) * S + k];
    spec[idx] = make_cuFloatComplex(a.x * b.x + a.y * b.y, a.y * b.x - a.x * b.y);
}

/// the normalized correlation surfaces, in place (M x M of each L x L correlation)
__global__ void normalize(Grid g, const int64_t *sat, const int64_t *sat2, int q0, int count, float *buf)
{
    const size_t MM = size_t(g.M) * g.M;
    const size_t idx = size_t(blockIdx.x) * blockDim.x + threadIdx.x;
    if (idx >= count * MM) return;
    const int p = int(idx / MM), k = int(idx % MM), u = k / g.M, v = k % g.M;
    const int q = q0 + p, f = q / g.nwin;
    int sy, sx, gy, gx;
    if (!pairWindow(g, q, sy, sx, gy, gx)) return;
    const int64_t stride = g.w + 1;
    float &c = buf[size_t(p) * g.L * g.L + u * g.L + v];
    c = normalized(c / (float(g.L) * g.L), g.norm[q % g.nwin],
                   boxSum(sat + f * g.satSize, stride, sy + u, sx + v, g.W),
                   boxSum(sat2 + f * g.satSize, stride, sy + u, sx + v, g.W), g.W * g.W);
}

/// the peak of each surface and the outputs; one block per pair
__global__ void peaks(Grid g, const float *buf, int q0, float *offset, float *peak, float *snr, float *cov)
{
    __shared__ float value[THREADS];
    __shared__ int where[THREADS];
    const int p = blockIdx.x, q = q0 + p;
    int sy, sx, gy, gx;
    if (!pairWindow(g, q, sy, sx, gy, gx)) {
        if (threadIdx.x == 0) invalidate(q, offset, peak, snr, cov);
        return;
    }
    const float *s = buf + size_t(p) * g.L * g.L;
    const int MM = g.M * g.M;
    float best = -2.0f;
    int at = MM;
    for (int k = threadIdx.x; k < MM; k += blockDim.x) {
        const float c = s[(k / g.M) * g.L + k % g.M];
        if (c > best) {
            best = c;
            at = k;
        }
    }
    value[threadIdx.x] = best;
    where[threadIdx.x] = at;
    __syncthreads();
    // the largest value, the first in the surface if equal (as the CPU backend)
    for (int half = blockDim.x / 2; half > 0; half /= 2) {
        if (threadIdx.x < half) {
            const float b = value[threadIdx.x + half];
            const int ib = where[threadIdx.x + half];
            if (b > value[threadIdx.x] || (b == value[threadIdx.x] && ib < where[threadIdx.x])) {
                value[threadIdx.x] = b;
                where[threadIdx.x] = ib;
            }
        }
        __syncthreads();
    }
    if (threadIdx.x != 0) return;
    if (!(value[0] > 0)) {
        invalidate(q, offset, peak, snr, cov);
        return;
    }
    writePeak(s, g.L, g.M, where[0] / g.M, where[0] % g.M, g.R, gy, gx, g.stat, g.W * g.W, q,
              offset, peak, snr, cov);
}

/// the device memory, freed when destroyed
class DeviceMemory {
public:
    ~DeviceMemory()
    {
        for (void *ptr : ptrs) cudaFree(ptr);
    }
    template <class T>
    T *alloc(size_t n)
    {
        void *ptr = nullptr;
        CHECK(cudaMalloc(&ptr, std::max<size_t>(n, 1) * sizeof(T)));
        ptrs.push_back(ptr);
        return static_cast<T *>(ptr);
    }

private:
    std::vector<void *> ptrs;
};

/// the cuFFT plans (forward, inverse) of the batch sizes used, destroyed with the object
class Plans {
public:
    ~Plans()
    {
        for (auto &kv : plans) {
            cufftDestroy(kv.second.first);
            cufftDestroy(kv.second.second);
        }
    }
    std::pair<cufftHandle, cufftHandle> get(int L, int count, cudaStream_t stream)
    {
        auto it = plans.find(count);
        if (it != plans.end()) return it->second;
        int n[2] = {L, L};
        cufftHandle fwd, inv;
        CHECK(cufftPlanMany(&fwd, 2, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_R2C, count));
        CHECK(cufftPlanMany(&inv, 2, n, nullptr, 1, 0, nullptr, 1, 0, CUFFT_C2R, count));
        CHECK(cufftSetStream(fwd, stream));
        CHECK(cufftSetStream(inv, stream));
        return plans[count] = {fwd, inv};
    }

private:
    std::map<int, std::pair<cufftHandle, cufftHandle>> plans;
};

class Cuda final : public Correlator {
public:
    Cuda(const Params &params, int dev) : p(params), device(dev)
    {
        CHECK(cudaSetDevice(device));
        CHECK(cudaStreamCreate(&stream));
        g.h = p.height;
        g.w = p.width;
        g.W = p.window;
        g.R = p.search;
        g.L = p.searchSize();
        g.Lh = g.L / 2 + 1;
        g.M = p.surfaceSize();
        g.nx = int(p.across.size());
        g.nwin = p.windows();
        g.stat = p.statSize;
        g.satSize = size_t(g.h + 1) * (g.w + 1);
        batch = p.batch;
        // the frames of a group: enough to fill a batch, at most 16
        group = std::clamp(batch / g.nwin, 1, 16);

        int *down = mem.alloc<int>(p.down.size()), *across = mem.alloc<int>(p.across.size());
        CHECK(cudaMemcpy(down, p.down.data(), p.down.size() * sizeof(int), cudaMemcpyHostToDevice));
        CHECK(cudaMemcpy(across, p.across.data(), p.across.size() * sizeof(int), cudaMemcpyHostToDevice));
        g.down = down;
        g.across = across;
        norm = mem.alloc<float>(g.nwin);
        g.norm = norm;
        S = size_t(g.L) * g.Lh;
        ref = mem.alloc<cufftComplex>(g.nwin * S);
        buf = mem.alloc<float>(size_t(batch) * g.L * g.L);
        spec = mem.alloc<cufftComplex>(batch * S);
        frames = mem.alloc<uint8_t>(size_t(group) * g.h * g.w);
        sat = mem.alloc<int64_t>(group * g.satSize);
        sat2 = mem.alloc<int64_t>(group * g.satSize);
        gross = mem.alloc<int>(size_t(group) * g.nwin * 2);
        offset = mem.alloc<float>(size_t(group) * g.nwin * 2);
        peak = mem.alloc<float>(size_t(group) * g.nwin);
        snr = mem.alloc<float>(size_t(group) * g.nwin);
        cov = mem.alloc<float>(size_t(group) * g.nwin * 3);
    }

    ~Cuda() override
    {
        if (stream) cudaStreamDestroy(stream);
    }

    Cuda(const Cuda &) = delete;
    Cuda &operator=(const Cuda &) = delete;

    std::string backend() const override { return "cuda"; }

    void setReference(const uint8_t *frame) override
    {
        CHECK(cudaSetDevice(device));
        const int chunk = std::min({batch, g.nwin, 512});
        const size_t LL = size_t(g.L) * g.L;
        std::vector<float> padded(chunk * LL), norms(g.nwin);
        for (int k0 = 0; k0 < g.nwin; k0 += chunk) {
            const int count = std::min(chunk, g.nwin - k0);
            referenceWindows(p, frame, k0, count, padded.data(), norms.data() + k0);
            CHECK(cudaMemcpyAsync(buf, padded.data(), count * LL * sizeof(float), cudaMemcpyHostToDevice, stream));
            CHECK(cufftExecR2C(plans.get(g.L, count, stream).first, buf, ref + k0 * S));
            CHECK(cudaStreamSynchronize(stream));
        }
        CHECK(cudaMemcpy(norm, norms.data(), g.nwin * sizeof(float), cudaMemcpyHostToDevice));
    }

    void correlate(const uint8_t *host, int n, const int *hostGross, const Outputs &out) override
    {
        CHECK(cudaSetDevice(device));
        const size_t pixels = size_t(g.h) * g.w;
        for (int f0 = 0; f0 < n; f0 += group) {
            const int nf = std::min(group, n - f0), pairs = nf * g.nwin;
            CHECK(cudaMemcpyAsync(frames, host + f0 * pixels, nf * pixels, cudaMemcpyHostToDevice, stream));
            Grid gg = g;
            gg.gross = nullptr;
            if (hostGross) {
                CHECK(cudaMemcpyAsync(gross, hostGross + size_t(f0) * g.nwin * 2, size_t(pairs) * 2 * sizeof(int),
                                      cudaMemcpyHostToDevice, stream));
                gg.gross = gross;
            }
            // the summed-area tables (the first row zero)
            CHECK(cudaMemsetAsync(sat, 0, nf * g.satSize * sizeof(int64_t), stream));
            CHECK(cudaMemsetAsync(sat2, 0, nf * g.satSize * sizeof(int64_t), stream));
            satRows<<<dim3(g.h, nf), THREADS, 0, stream>>>(frames, g.h, g.w, g.satSize, sat, sat2);
            satColumns<<<dim3(unsigned(divUp(g.w, THREADS)), nf), THREADS, 0, stream>>>(g.h, g.w, g.satSize, sat, sat2);
            CHECK(cudaGetLastError());

            for (int q0 = 0; q0 < pairs; q0 += batch) {
                const int count = std::min(batch, pairs - q0);
                const auto plan = plans.get(g.L, count, stream);
                const size_t LL = size_t(g.L) * g.L, MM = size_t(g.M) * g.M;
                extract<<<unsigned(divUp(count * LL, THREADS)), THREADS, 0, stream>>>(gg, frames, sat, q0, count, buf);
                CHECK(cudaGetLastError());
                CHECK(cufftExecR2C(plan.first, buf, spec));
                multiplyConjugate<<<unsigned(divUp(count * S, THREADS)), THREADS, 0, stream>>>(
                    spec, ref, q0, count, g.nwin, int(S));
                CHECK(cudaGetLastError());
                CHECK(cufftExecC2R(plan.second, spec, buf));
                normalize<<<unsigned(divUp(count * MM, THREADS)), THREADS, 0, stream>>>(gg, sat, sat2, q0, count, buf);
                peaks<<<count, THREADS, 0, stream>>>(gg, buf, q0, offset, peak, snr, cov);
                CHECK(cudaGetLastError());
            }
            const size_t first = size_t(f0) * g.nwin;
            CHECK(cudaMemcpyAsync(out.offset + 2 * first, offset, pairs * 2 * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(out.peak + first, peak, pairs * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(out.snr + first, snr, pairs * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CHECK(cudaMemcpyAsync(out.cov + 3 * first, cov, pairs * 3 * sizeof(float), cudaMemcpyDeviceToHost, stream));
            CHECK(cudaStreamSynchronize(stream));
        }
    }

private:
    Params p;
    int device;
    cudaStream_t stream = nullptr;
    DeviceMemory mem;
    Plans plans;
    Grid g{};
    int batch = 1, group = 1;
    size_t S = 0;                     // the size of a spectrum, L x (L / 2 + 1)
    float *norm = nullptr;            // the norms of the zero-mean reference windows
    cufftComplex *ref = nullptr;      // the spectra of the reference windows
    float *buf = nullptr;             // a batch of search windows, then of surfaces
    cufftComplex *spec = nullptr;     // their spectra
    uint8_t *frames = nullptr;        // a group of frames
    int64_t *sat = nullptr, *sat2 = nullptr;   // their summed-area tables
    int *gross = nullptr;             // their gross offsets
    float *offset = nullptr, *peak = nullptr, *snr = nullptr, *cov = nullptr;   // their outputs
};

} // namespace

bool hasCuda()
{
    int count = 0;
    return cudaGetDeviceCount(&count) == cudaSuccess && count > 0;
}

std::unique_ptr<Correlator> makeCuda(const Params &p, int device)
{
    checkParams(p);
    return std::make_unique<Cuda>(p, device);
}

} // namespace framecor
// end of file
