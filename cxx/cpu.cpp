/**
 * @file  cpu.cpp
 * @brief The CPU backend: FFTW, the windows of a frame in parallel with OpenMP
 */

#include <fftw3.h>
#ifdef _OPENMP
#include <omp.h>
#endif

#include <algorithm>
#include <complex>
#include <cstring>
#include <stdexcept>
#include <vector>

#include "framecor.h"
#include "surface.h"

namespace framecor {
namespace {

/// the summed-area tables of a frame and of its squares, (height + 1) x (width + 1)
void summedArea(const uint8_t *frame, int h, int w, int64_t *sat, int64_t *sat2)
{
    const size_t stride = w + 1;
    std::fill(sat, sat + stride, 0);
    std::fill(sat2, sat2 + stride, 0);
    for (int y = 0; y < h; y++) {
        const uint8_t *row = frame + size_t(y) * w;
        int64_t *s = sat + (y + 1) * stride, *s2 = sat2 + (y + 1) * stride;
        const int64_t *above = s - stride, *above2 = s2 - stride;
        int64_t a = 0, b = 0;
        s[0] = s2[0] = 0;
        for (int x = 0; x < w; x++) {
            a += row[x];
            b += int64_t(row[x]) * row[x];
            s[x + 1] = above[x + 1] + a;
            s2[x + 1] = above2[x + 1] + b;
        }
    }
}

int threadNumber()
{
#ifdef _OPENMP
    return omp_get_thread_num();
#else
    return 0;
#endif
}

class Cpu final : public Correlator {
public:
    Cpu(const Params &params, int threads)
        : p(params), L(p.searchSize()), Lh(L / 2 + 1), M(p.surfaceSize()), nx(int(p.across.size())),
          nwin(p.windows()), stride(p.width + 1)
    {
#ifdef _OPENMP
        nthreads = threads > 0 ? threads : omp_get_max_threads();
#else
        nthreads = 1;
#endif
        for (int t = 0; t < nthreads; t++) {
            real.push_back(fftwf_alloc_real(size_t(L) * L));
            spec.push_back(fftwf_alloc_complex(size_t(L) * Lh));
            if (!real.back() || !spec.back()) throw std::bad_alloc();
        }
        // the plans, executed on the buffers of each thread (all aligned by fftwf_alloc)
        fwd = fftwf_plan_dft_r2c_2d(L, L, real[0], spec[0], FFTW_MEASURE);
        inv = fftwf_plan_dft_c2r_2d(L, L, spec[0], real[0], FFTW_MEASURE);
        if (!fwd || !inv) throw std::runtime_error("framecor: FFTW planning failed");
        ref.resize(size_t(nwin) * L * Lh);
        norm.resize(nwin);
        sat.resize(size_t(p.height + 1) * stride);
        sat2.resize(sat.size());
    }

    ~Cpu() override
    {
        if (fwd) fftwf_destroy_plan(fwd);
        if (inv) fftwf_destroy_plan(inv);
        for (float *b : real) fftwf_free(b);
        for (fftwf_complex *b : spec) fftwf_free(b);
    }

    Cpu(const Cpu &) = delete;
    Cpu &operator=(const Cpu &) = delete;

    std::string backend() const override { return "cpu"; }

    void setReference(const uint8_t *frame) override
    {
        const size_t S = size_t(L) * Lh;
        #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 8)
        for (int k = 0; k < nwin; k++) {
            const int t = threadNumber();
            norm[k] = referenceWindow(p, frame, k, real[t]);
            fftwf_execute_dft_r2c(fwd, real[t], spec[t]);
            std::memcpy(ref.data() + k * S, spec[t], S * sizeof(fftwf_complex));
        }
    }

    void correlate(const uint8_t *frames, int n, const int *gross, const Outputs &out) override
    {
        for (int f = 0; f < n; f++) {
            const uint8_t *frame = frames + size_t(f) * p.height * p.width;
            summedArea(frame, p.height, p.width, sat.data(), sat2.data());
            #pragma omp parallel for num_threads(nthreads) schedule(dynamic, 8)
            for (int k = 0; k < nwin; k++)
                window(frame, int64_t(f) * nwin + k, k, gross, out);
        }
    }

private:
    /// correlate the window k of a frame; q, its index in the outputs (and in gross)
    void window(const uint8_t *frame, int64_t q, int k, const int *gross, const Outputs &out)
    {
        const int gy = gross ? gross[2 * q] : 0, gx = gross ? gross[2 * q + 1] : 0;
        int sy, sx;
        if (norm[k] <= 0 || !searchWindow(p.down[k / nx], p.across[k % nx], gy, gx, p.search, L,
                                          p.height, p.width, sy, sx)) {
            invalidate(q, out.offset, out.peak, out.snr, out.cov);
            return;
        }
        const int t = threadNumber();
        float *buf = real[t];
        auto *sp = reinterpret_cast<std::complex<float> *>(spec[t]);

        // the search window, its mean subtracted (which leaves the correlation with the zero-mean
        // reference window unchanged, and keeps the FFT accurate)
        const float mean = float(boxSum(sat.data(), stride, sy, sx, L)) / (float(L) * L);
        for (int i = 0; i < L; i++) {
            const uint8_t *row = frame + size_t(sy + i) * p.width + sx;
            for (int j = 0; j < L; j++)
                buf[i * L + j] = row[j] - mean;
        }
        // the correlation with the reference window
        fftwf_execute_dft_r2c(fwd, buf, spec[t]);
        const std::complex<float> *r = ref.data() + size_t(k) * L * Lh;
        for (size_t i = 0; i < size_t(L) * Lh; i++)
            sp[i] *= std::conj(r[i]);
        fftwf_execute_dft_c2r(inv, spec[t], buf);

        // normalized, in place (row stride L), and its peak
        const float scale = 1.0f / (float(L) * L);
        const int n = p.window * p.window;
        float best = -2.0f;
        int bu = 0, bv = 0;
        for (int u = 0; u < M; u++)
            for (int v = 0; v < M; v++) {
                float &c = buf[u * L + v];
                c = normalized(c * scale, norm[k], boxSum(sat.data(), stride, sy + u, sx + v, p.window),
                               boxSum(sat2.data(), stride, sy + u, sx + v, p.window), n);
                if (c > best) {
                    best = c;
                    bu = u;
                    bv = v;
                }
            }
        if (!(best > 0)) {
            invalidate(q, out.offset, out.peak, out.snr, out.cov);
            return;
        }
        writePeak(buf, L, M, bu, bv, p.search, gy, gx, p.statSize, n, q, out.offset, out.peak,
                  out.snr, out.cov);
    }

    Params p;
    int L, Lh, M, nx, nwin;
    int64_t stride;
    int nthreads = 1;
    std::vector<float *> real;              // per thread: the search window, then the surface
    std::vector<fftwf_complex *> spec;      // per thread: its spectrum
    fftwf_plan fwd = nullptr, inv = nullptr;
    std::vector<std::complex<float>> ref;   // the spectra of the reference windows
    std::vector<float> norm;                // the norms of the zero-mean reference windows
    std::vector<int64_t> sat, sat2;         // the summed-area tables of the frame
};

} // namespace

std::unique_ptr<Correlator> makeCpu(const Params &p, int threads)
{
    checkParams(p);
    return std::make_unique<Cpu>(p, threads);
}

} // namespace framecor
// end of file
