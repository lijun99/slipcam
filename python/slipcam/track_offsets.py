#!/usr/bin/env python3
"""
Dense offsets of the frames of a video relative to a reference frame, with framecor

Each frame from --first to --last is correlated with the reference frame (--ref, e.g., the last one
before the shaking), on a grid of windows; the offsets are in pixels, (down, across), the motion of
the frame relative to the reference. The windows that overlap the burned-in text of the camera
(--mask, boxes in pixels) are flagged, as the text does not move with the scene.

The correlation is framecor's (the frames in memory, the reference windows transformed once), on the
GPU, or on the CPU with --cpu.

usage:
  slipcam-track run/frames.npy --ref 285 --first 140 --out run/offsets.npz \
      --mask 870 25 1240 68 --mask 40 650 630 700
needs: numpy
"""

import argparse
import time

import numpy as np

from . import framecor


def boxes_mask(y0, x0, window, boxes):
    """The windows that overlap any of the boxes (x0, y0, x1, y1)"""
    masked = np.zeros((len(y0), len(x0)), bool)
    for bx0, by0, bx1, by1 in boxes:
        masked |= ((y0[:, None] < by1) & (y0[:, None] + window > by0) & (x0[None, :] < bx1) & (x0[None, :] + window > bx0))
    return masked


def track(frames, ref, index, masked=None, window=64, skip=16, search=48, cpu=False, chunk=64):
    """The offsets of the frames index relative to the frame ref, chunk frames per call of framecor; a dict
    as saved by main"""
    n, h, w = frames.shape
    y0, x0 = framecor.window_grid(h, w, window, search, skip)
    ny, nx = len(y0), len(x0)
    if masked is None:
        masked = np.zeros((ny, nx), bool)
    cor = framecor.Correlator(h, w, y0, x0, window=window, search=search, backend='cpu' if cpu else 'auto')
    index = np.asarray(index)
    out = {key: np.empty((len(index), ny, nx) + tail, np.float32)
           for key, tail in (('offsets', (2,)), ('snr', ()), ('cov', (3,)), ('peak', ()))}
    tic = time.perf_counter()
    cor.set_reference(np.ascontiguousarray(frames[ref]))
    for k0 in range(0, len(index), chunk):
        sel = index[k0:k0 + chunk]
        # a run of consecutive frames: a slice of the (memory-mapped) frames, without a copy
        run = np.array_equal(sel, np.arange(sel[0], sel[0] + len(sel)))
        r = cor.correlate(np.ascontiguousarray(frames[sel[0]:sel[0] + len(sel)] if run else frames[sel]))
        for key in out:
            out[key][k0:k0 + len(sel)] = r[key]
    elapsed = time.perf_counter() - tic
    print(f'{len(index)} frames of {ny} x {nx} windows in {elapsed:.1f} s ({elapsed / len(index) * 1e3:.1f} ms per frame, '
          f'framecor {cor.backend}), {masked.sum()} windows masked')
    return dict(index=index, ref=ref, masked=masked, yc=y0 + window / 2, xc=x0 + window / 2, window=window,
                skip=skip, search=search, engine='framecor', **out)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('frames', help='the .npy of the frames (frames x height x width, uint8)')
    parser.add_argument('--ref', type=int, required=True, help='the reference frame')
    parser.add_argument('--first', type=int, default=0, help='the first frame (default: %(default)s)')
    parser.add_argument('--last', type=int, help='the last frame (default: the last of the video)')
    parser.add_argument('--window', type=int, default=64, help='the window size, pixels (default: %(default)s)')
    parser.add_argument('--skip', type=int, default=16, help='the distance between windows, pixels (default: %(default)s)')
    parser.add_argument('--search', type=int, default=48, help='the half search range, pixels (default: %(default)s)')
    parser.add_argument('--mask', type=int, nargs=4, action='append', default=[], metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='a box of burned-in text, pixels (repeatable)')
    parser.add_argument('--cpu', action='store_true', help='use the CPU implementation')
    parser.add_argument('--out', required=True, help='the output .npz')
    args = parser.parse_args()

    frames = np.load(args.frames, mmap_mode='r')
    n, h, w = frames.shape
    last = n - 1 if args.last is None else args.last
    y0, x0 = framecor.window_grid(h, w, args.window, args.search, args.skip)
    masked = boxes_mask(y0, x0, args.window, args.mask)
    result = track(frames, args.ref, np.arange(args.first, last + 1), masked=masked,
                   window=args.window, skip=args.skip, search=args.search, cpu=args.cpu)
    np.savez(args.out, **result)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
