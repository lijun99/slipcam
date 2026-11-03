/**
 * @file  surface.h
 * @brief The normalized correlation surface and its peak, shared by the CPU and CUDA backends
 */

#pragma once

#include <math.h>
#include <stdint.h>

#ifdef __CUDACC__
#define FRAMECOR_HD __host__ __device__
#else
#define FRAMECOR_HD
#endif

namespace framecor {

/// the sum over the box of n x n pixels with first pixel (y, x), from a summed-area table with a row
/// stride `stride` (the table has a first row and column of zeros)
FRAMECOR_HD inline int64_t boxSum(const int64_t *sat, int64_t stride, int y, int x, int n)
{
    return sat[(y + n) * stride + x + n] - sat[y * stride + x + n] - sat[(y + n) * stride + x]
        + sat[y * stride + x];
}

/// the normalized correlation: corr, the sum of the zero-mean reference window times the search
/// window at a shift; norm, the norm of the zero-mean reference window; s, s2, the sum and the sum
/// of squares of the n pixels of the search window at the shift
FRAMECOR_HD inline float normalized(float corr, float norm, int64_t s, int64_t s2, int n)
{
    const int64_t v = n * s2 - s * s;   // n^2 times the variance
    if (v <= 0 || norm <= 0) return 0.0f;
    return corr / (norm * sqrtf(float(v) / float(n)));
}

/// the first pixel (sy, sx) of the search window of the reference window at (y0, x0), with the gross
/// offset (gy, gx), the half search range R and the search window size L; whether it is in the frame
FRAMECOR_HD inline bool searchWindow(int y0, int x0, int gy, int gx, int R, int L, int h, int w,
                                     int &sy, int &sx)
{
    sy = y0 + gy - R;
    sx = x0 + gx - R;
    return sy >= 0 && sx >= 0 && sy + L <= h && sx + L <= w;
}

/// the outputs of a window that could not be correlated
FRAMECOR_HD inline void invalidate(int64_t q, float *offset, float *peak, float *snr, float *cov)
{
    offset[2 * q] = offset[2 * q + 1] = nanf("");
    peak[q] = snr[q] = 0.0f;
    cov[3 * q] = cov[3 * q + 1] = 99.0f;
    cov[3 * q + 2] = 0.0f;
}

/// the sub-pixel offset of the peak of a parabola through (-1, m), (0, c), (1, p)
FRAMECOR_HD inline float parabola(float m, float c, float p)
{
    const float den = m - 2 * c + p;
    return den < 0 ? 0.5f * (m - p) / den : 0.0f;
}

/**
 * The outputs of the window q from its normalized correlation surface
 * @param s the surface, M x M, with a row stride `stride`
 * @param u, v the integer peak (down, across)
 * @param R the half search range; (gy, gx) the gross offset
 * @param statSize the size of the surface around the peak for the SNR
 * @param templateSize the number of pixels of the reference window, for the covariance
 */
FRAMECOR_HD inline void writePeak(const float *s, int stride, int M, int u, int v, int R, int gy,
                                  int gx, int statSize, int templateSize, int64_t q, float *offset,
                                  float *peak, float *snr, float *cov)
{
    const float f0 = s[u * stride + v];
    const bool inside = u > 0 && u < M - 1 && v > 0 && v < M - 1;

    // the sub-pixel peak: a quadratic fitted to the 3 x 3 surface; else a parabola along each axis
    float du = 0.0f, dv = 0.0f;
    if (inside) {
        float f[3][3];
        for (int i = 0; i < 3; i++)
            for (int j = 0; j < 3; j++)
                f[i][j] = s[(u + i - 1) * stride + v + j - 1];
        const float c1 = (f[2][0] + f[2][1] + f[2][2] - f[0][0] - f[0][1] - f[0][2]) / 6;
        const float c2 = (f[0][2] + f[1][2] + f[2][2] - f[0][0] - f[1][0] - f[2][0]) / 6;
        const float c3 = (f[0][0] + f[0][1] + f[0][2] + f[2][0] + f[2][1] + f[2][2]
                          - 2 * (f[1][0] + f[1][1] + f[1][2])) / 6;
        const float c5 = (f[0][0] + f[1][0] + f[2][0] + f[0][2] + f[1][2] + f[2][2]
                          - 2 * (f[0][1] + f[1][1] + f[2][1])) / 6;
        const float c4 = (f[2][2] - f[2][0] - f[0][2] + f[0][0]) / 4;
        const float det = 4 * c3 * c5 - c4 * c4;
        bool fitted = false;
        if (c3 < 0 && det > 0) {
            du = (c4 * c2 - 2 * c5 * c1) / det;
            dv = (c4 * c1 - 2 * c3 * c2) / det;
            fitted = fabsf(du) <= 1 && fabsf(dv) <= 1;
        }
        if (!fitted) {
            du = parabola(f[0][1], f[1][1], f[2][1]);
            dv = parabola(f[1][0], f[1][1], f[1][2]);
        }
    }
    offset[2 * q] = float(gy + u - R) + du;
    offset[2 * q + 1] = float(gx + v - R) + dv;
    peak[q] = f0;

    // the SNR: the squared peak over the mean of the squared surface around it, without the peak
    const int n = statSize < M ? statSize : M;
    double sum = 0.0;
    int count = 0;
    for (int a = 0; a < n; a++) {
        const int i = u + a - n / 2;
        if (i < 0 || i >= M) continue;
        for (int b = 0; b < n; b++) {
            const int j = v + b - n / 2;
            if (j < 0 || j >= M) continue;
            const double c = s[i * stride + j];
            sum += c * c;
            count++;
        }
    }
    const double mean = count > 1 ? (sum - double(f0) * f0) / (count - 1) : 0.0;
    snr[q] = mean > 0 ? float(double(f0) * f0 / mean) : 0.0f;

    // the covariance, from the curvature of the surface at the peak
    cov[3 * q] = cov[3 * q + 1] = 99.0f;
    cov[3 * q + 2] = 0.0f;
    if (inside) {
        const double T = templateSize;
        const double dxx = -(s[(u + 1) * stride + v] + s[(u - 1) * stride + v] - 2.0 * f0) * T;
        const double dyy = -(s[u * stride + v + 1] + s[u * stride + v - 1] - 2.0 * f0) * T;
        const double dxy = (s[(u + 1) * stride + v + 1] + s[(u - 1) * stride + v - 1]
                            - s[(u + 1) * stride + v - 1] - s[(u - 1) * stride + v + 1]) * 0.25 * T;
        double n2 = 1.0 - f0 > 0 ? 1.0 - f0 : 0.0;
        const double n4 = n2 * n2 * 0.5 * T;
        n2 *= 2;
        const double w = dxy * dxy - dxx * dyy;
        if (fabs(w) >= 1e-2) {
            const double w2 = w * w;
            cov[3 * q] = float((-n2 * w * dyy + n4 * (dyy * dyy + dxy * dxy)) / w2);
            cov[3 * q + 1] = float((-n2 * w * dxx + n4 * (dxx * dxx + dxy * dxy)) / w2);
            cov[3 * q + 2] = float((n2 * w * dxy - n4 * (dxx + dyy) * dxy) / w2);
        }
    }
}

} // namespace framecor
// end of file
