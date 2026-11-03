"""
framecor: dense normalized cross-correlation of video frames with a reference frame, on the GPU
(CUDA, cuFFT) or the CPU (FFTW, OpenMP)

  grid = framecor.window_grid(720, 1280, window=64, search=48, skip=16)
  cor = framecor.Correlator(720, 1280, *grid, window=64, search=48)
  cor.set_reference(frames[ref])
  result = cor.correlate(frames[first:last])     # offsets, peak, snr, cov
"""

import numpy as np

from ._framecor import Correlator, has_cuda, __version__

__all__ = ['Correlator', 'has_cuda', 'window_grid', '__version__']


def window_grid(height, width, window, search, skip, margin=0):
    """
    The first pixels (down, across) of the grid of windows whose search windows are in the frame,
    as pycuampcor.window_grid: from margin + search, (size - 2 (margin + search) - window) // skip
    windows in each direction; a ValueError if no window fits
    """
    axes = []
    for direction, size in (('down', height), ('across', width)):
        first = margin + search
        count = (size - margin - search - first - window) // skip
        if count <= 0:
            raise ValueError(f'window_grid: no window fits {direction}: frame size {size}, window {window}, '
                             f'search range {search}, margin {margin}')
        axes.append(first + skip * np.arange(count))
    return tuple(axes)
