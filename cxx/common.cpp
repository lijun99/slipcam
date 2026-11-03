/**
 * @file  common.cpp
 * @brief The parameters and the reference windows, shared by the backends
 */

#include <algorithm>
#include <cmath>
#include <stdexcept>
#include <string>

#include "framecor.h"

namespace framecor {

void checkParams(const Params &p)
{
    auto fail = [](const std::string &msg) { throw std::invalid_argument("framecor: " + msg); };
    if (p.height <= 0 || p.width <= 0) fail("the frame size must be positive");
    if (p.window < 3) fail("the window must be at least 3 pixels");
    if (p.search < 1) fail("the half search range must be at least 1 pixel");
    if (p.down.empty() || p.across.empty()) fail("no windows");
    for (int y : p.down)
        if (y < 0 || y + p.window > p.height) fail("a window is outside the frame (down " + std::to_string(y) + ")");
    for (int x : p.across)
        if (x < 0 || x + p.window > p.width) fail("a window is outside the frame (across " + std::to_string(x) + ")");
    if (p.statSize < 3) fail("the SNR window must be at least 3 pixels");
    if (p.batch < 1) fail("the batch must be at least 1 window");
}

float referenceWindow(const Params &p, const uint8_t *frame, int index, float *padded)
{
    const int L = p.searchSize(), W = p.window;
    const int y0 = p.down[index / p.across.size()], x0 = p.across[index % p.across.size()];
    double sum = 0.0;
    for (int i = 0; i < W; i++)
        for (int j = 0; j < W; j++)
            sum += frame[size_t(y0 + i) * p.width + x0 + j];
    const double mean = sum / (double(W) * W);
    std::fill(padded, padded + size_t(L) * L, 0.0f);
    double sq = 0.0;
    for (int i = 0; i < W; i++)
        for (int j = 0; j < W; j++) {
            const float d = float(frame[size_t(y0 + i) * p.width + x0 + j] - mean);
            padded[i * L + j] = d;
            sq += double(d) * d;
        }
    return float(std::sqrt(sq));
}

void referenceWindows(const Params &p, const uint8_t *frame, int first, int count, float *padded, float *norm)
{
    const size_t LL = size_t(p.searchSize()) * p.searchSize();
    #pragma omp parallel for schedule(static)
    for (int k = 0; k < count; k++)
        norm[k] = referenceWindow(p, frame, first + k, padded + k * LL);
}

#ifndef FRAMECOR_WITH_CUDA
bool hasCuda() { return false; }

std::unique_ptr<Correlator> makeCuda(const Params &, int)
{
    throw std::runtime_error("framecor: the CUDA backend is not built");
}
#endif

} // namespace framecor
// end of file
