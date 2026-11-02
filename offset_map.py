#!/usr/bin/env python3
"""
Map of the offsets of the video frames at a given time, over the reference frame

The offsets (from track_offsets.py) are the medians over the frames --frames FIRST LAST, of the
windows with SNR above --min-snr; across (positive to the right) and down (positive down), in pixels,
drawn over the reference frame with a diverging color scale (zero is gray).

usage:
  offset_map.py run/offsets.npz run/frames.npy --frames 770 806 --out run/final_map.png
needs: numpy, matplotlib
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt                                     # noqa: E402
from matplotlib.colors import LinearSegmentedColormap              # noqa: E402

# the diverging pair of the reference palette: blue <-> red, gray midpoint
DIVERGING = LinearSegmentedColormap.from_list('blue_gray_red', ['#2a78d6', '#f0efec', '#e34948'])
INK, INK2 = '#0b0b0b', '#52514e'


def median_offsets(z, first, last, min_snr):
    """The median offsets (down, across) of the windows over the frames first..last"""
    sel = (z['index'] >= first) & (z['index'] <= last)
    off = z['offsets'][sel].copy()
    bad = (z['snr'][sel] < min_snr) | z['masked'][None]
    off[bad] = np.nan
    return np.nanmedian(off, axis=0)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('offsets', help='the .npz of track_offsets.py')
    parser.add_argument('frames', help='the .npy of the frames')
    parser.add_argument('--frames', dest='span', type=int, nargs=2, required=True, metavar=('FIRST', 'LAST'),
                        help='the frames to take the median over')
    parser.add_argument('--min-snr', type=float, default=5, help='the smallest SNR of the windows used (default: %(default)s)')
    parser.add_argument('--vmax', type=float, default=20, help='the limit of the color scale, pixels (default: %(default)s)')
    parser.add_argument('--out', required=True, help='the output image')
    args = parser.parse_args()

    z = np.load(args.offsets)
    med = median_offsets(z, *args.span, args.min_snr)
    ref = np.load(args.frames, mmap_mode='r')[int(z['ref'])]
    h, w = ref.shape
    skip, xc, yc = int(z['skip']), z['xc'], z['yc']
    extent = (xc[0] - skip / 2, xc[-1] + skip / 2, yc[-1] + skip / 2, yc[0] - skip / 2)

    fig, axes = plt.subplots(1, 2, figsize=(15, 4.9), constrained_layout=True)
    for ax, comp, name in zip(axes, (1, 0), ('across (+ right)', 'down (+ down)')):
        ax.imshow(ref, cmap='gray', vmin=0, vmax=255, extent=(0, w, h, 0))
        im = ax.imshow(med[..., comp], cmap=DIVERGING, vmin=-args.vmax, vmax=args.vmax, alpha=0.75,
                       extent=extent, interpolation='nearest')
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(f'Offset {name}, frames {args.span[0]}-{args.span[1]} vs reference {int(z["ref"])}',
                     color=INK, fontsize=11, loc='left')
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
        cb.set_label('pixels', color=INK2, fontsize=9)
        cb.ax.tick_params(colors=INK2, labelsize=8)
        cb.outline.set_visible(False)
    fig.savefig(args.out, dpi=110, facecolor='#fcfcfb')
    print(f'-> {args.out}')


if __name__ == '__main__':
    main()
