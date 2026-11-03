/**
 * @file  framecor.h
 * @brief Dense normalized cross-correlation of video frames with a reference frame
 *
 * A grid of square windows of the reference frame is searched in each frame within +/- search
 * pixels of its position (plus an optional gross offset per frame and window). The correlation is
 * computed by FFT, normalized with the summed-area tables of the frame (the frames are 8-bit, so
 * the sums are exact integers), and its peak refined to sub-pixel by a quadratic fit to the 3 x 3
 * surface around it. The SNR and the covariance of the offsets are as in pycuampcor.
 */

#pragma once

#include <cstdint>
#include <memory>
#include <string>
#include <vector>

namespace framecor {

/// the frames, the grid of the reference windows and the correlation parameters
struct Params {
    int height = 0;                 ///< the height of the frames
    int width = 0;                  ///< the width of the frames
    int window = 64;                ///< the size of the (square) reference windows
    int search = 48;                ///< the half search range, pixels
    std::vector<int> down;          ///< the first pixel (down) of each row of windows
    std::vector<int> across;        ///< the first pixel (across) of each column of windows
    int statSize = 21;              ///< the size of the surface around the peak for the SNR
    int batch = 4096;               ///< the windows correlated together (CUDA: the FFT batch)

    int windows() const { return int(down.size() * across.size()); }
    int searchSize() const { return window + 2 * search; }   ///< the size of the search windows
    int surfaceSize() const { return 2 * search + 1; }       ///< the size of the correlation surface
};

/// the outputs for n frames, n x windows (the windows of a row of the grid, then the next row)
struct Outputs {
    float *offset;   ///< n x windows x 2: (down, across), the motion of the frame; NaN if not valid
    float *peak;     ///< n x windows: the normalized correlation at the peak
    float *snr;      ///< n x windows: the squared peak over the mean squared surface around it
    float *cov;      ///< n x windows x 3: the covariance of the offsets (down, across, cross)
};

/// the correlator of frames with a reference frame, on a backend
class Correlator {
public:
    virtual ~Correlator() = default;
    /// set the reference frame (height x width), and compute the spectra of its windows
    virtual void setReference(const uint8_t *frame) = 0;
    /// correlate n frames (n x height x width) with the reference frame; gross, the integer
    /// offsets (down, across) of the search windows, n x windows x 2, or nullptr
    virtual void correlate(const uint8_t *frames, int n, const int *gross, const Outputs &out) = 0;
    /// the name of the backend
    virtual std::string backend() const = 0;
};

/// check the parameters; std::invalid_argument if not valid
void checkParams(const Params &p);
/// the CPU backend (FFTW, OpenMP); threads 0 for the default number of threads
std::unique_ptr<Correlator> makeCpu(const Params &p, int threads);
/// whether the CUDA backend is built and a GPU is available
bool hasCuda();
/// the CUDA backend; std::runtime_error if not built
std::unique_ptr<Correlator> makeCuda(const Params &p, int device);

// shared by the backends

/// the reference window index, zero-mean and zero-padded to searchSize x searchSize; its norm
float referenceWindow(const Params &p, const uint8_t *frame, int index, float *padded);
/// the reference windows first to first + count - 1 (count x searchSize x searchSize), and their norms
void referenceWindows(const Params &p, const uint8_t *frame, int first, int count, float *padded, float *norm);

} // namespace framecor
// end of file
