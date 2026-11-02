#!/usr/bin/env python3
"""
Dense offsets of the frames of a video relative to a reference frame, with pycuampcor

Each frame from --first to --last is correlated with the reference frame (--ref, e.g., the last one
before the shaking), on a grid of windows; the offsets are in pixels, (down, across), the motion of
the frame relative to the reference. The windows that overlap the burned-in text of the camera
(--mask, boxes in pixels) are flagged, as the text does not move with the scene.

usage:
  track_offsets.py run/frames.npy --ref 285 --first 140 --out run/offsets.npz \
      --mask 870 25 1240 68 --mask 40 650 630 700
needs: numpy, pycuampcor
"""

import argparse
import os
import time

import numpy as np
import pycuampcor


def window_grid(h, w, window, skip, search):
    """The grid of the windows of pycuampcor: its parameters, and the first pixels (down, across)"""
    grid = pycuampcor.window_grid(h, w, window=(window, window), search=(search, search), skip=(skip, skip))
    y0 = grid['referenceStartPixelDownStatic'] + skip * np.arange(grid['numberWindowDown'])
    x0 = grid['referenceStartPixelAcrossStatic'] + skip * np.arange(grid['numberWindowAcross'])
    return grid, y0, x0


def boxes_mask(y0, x0, window, boxes):
    """The windows that overlap any of the boxes (x0, y0, x1, y1)"""
    masked = np.zeros((len(y0), len(x0)), bool)
    for bx0, by0, bx1, by1 in boxes:
        masked |= ((y0[:, None] < by1) & (y0[:, None] + window > by0) & (x0[None, :] < bx1) & (x0[None, :] + window > bx0))
    return masked


def track(frames, ref, index, workdir, masked=None, window=64, skip=16, search=48, cpu=False):
    """The offsets of the frames index relative to the frame ref, with pycuampcor; a dict as saved by main"""
    n, h, w = frames.shape
    os.makedirs(workdir, exist_ok=True)
    refname, sec = os.path.join(workdir, 'reference.f32'), os.path.join(workdir, 'secondary.f32')
    out = {k: os.path.join(workdir, k + '.bip') for k in ('offset', 'snr', 'cov', 'gross', 'peak')}
    frames[ref].astype(np.float32).tofile(refname)

    grid, y0, x0 = window_grid(h, w, window, skip, search)
    ny, nx = len(y0), len(x0)
    if masked is None:
        masked = np.zeros((ny, nx), bool)

    params = dict(referenceImageName=refname, referenceImageHeight=h, referenceImageWidth=w,
                  referenceImageDataType='float32',
                  secondaryImageName=sec, secondaryImageHeight=h, secondaryImageWidth=w,
                  secondaryImageDataType='float32',
                  windowSizeHeight=window, windowSizeWidth=window, halfSearchRangeDown=search,
                  halfSearchRangeAcross=search, skipSampleDown=skip, skipSampleAcross=skip, **grid,
                  derampMethod='magnitude', deviceID=0, nStreams=2,
                  offsetImageName=out['offset'], snrImageName=out['snr'], covImageName=out['cov'],
                  grossOffsetImageName=out['gross'], peakValueImageName=out['peak'])
    pycuampcor.set_log_level(pycuampcor.LogLevel.WARNING)

    index = np.asarray(index)
    offsets = np.full((len(index), ny, nx, 2), np.nan, np.float32)
    snr = np.zeros((len(index), ny, nx), np.float32)
    cov = np.zeros((len(index), ny, nx, 3), np.float32)
    tic = time.perf_counter()
    for k, i in enumerate(index):
        frames[i].astype(np.float32).tofile(sec)
        ampcor = pycuampcor.create(use_gpu=not cpu, **params)
        ampcor.runAmpcor()
        del ampcor
        offsets[k] = np.fromfile(out['offset'], np.float32).reshape(ny, nx, 2)
        snr[k] = np.fromfile(out['snr'], np.float32).reshape(ny, nx)
        cov[k] = np.fromfile(out['cov'], np.float32).reshape(ny, nx, 3)
        if k % 100 == 0:
            print(f'frame {i} ({k + 1} of {len(index)}), {time.perf_counter() - tic:.1f} s')
    elapsed = time.perf_counter() - tic
    print(f'{len(index)} frames of {ny} x {nx} windows in {elapsed:.1f} s ({elapsed / len(index) * 1e3:.0f} ms per frame), '
          f'{masked.sum()} windows masked')
    return dict(index=index, ref=ref, offsets=offsets, snr=snr, cov=cov, masked=masked,
                yc=y0 + window / 2, xc=x0 + window / 2, window=window, skip=skip, search=search)


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
    parser.add_argument('--workdir', help='the directory of the temporary images (default: next to --out)')
    parser.add_argument('--out', required=True, help='the output .npz')
    args = parser.parse_args()

    frames = np.load(args.frames, mmap_mode='r')
    n, h, w = frames.shape
    last = n - 1 if args.last is None else args.last
    workdir = args.workdir or os.path.join(os.path.dirname(os.path.abspath(args.out)), 'ampcor')
    _, y0, x0 = window_grid(h, w, args.window, args.skip, args.search)
    masked = boxes_mask(y0, x0, args.window, args.mask)
    result = track(frames, args.ref, np.arange(args.first, last + 1), workdir, masked=masked,
                   window=args.window, skip=args.skip, search=args.search, cpu=args.cpu)
    np.savez(args.out, **result)
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
