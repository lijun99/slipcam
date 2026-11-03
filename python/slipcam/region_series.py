#!/usr/bin/env python3
"""
Time series of the offsets of regions of the video frames

For each region (--region NAME X0 Y0 X1 Y1, pixels of the frame), the median offsets (across,
down) of its windows with SNR above --min-snr, per frame; the time is relative to the reference frame,
at --fps. The series are saved to a .csv and plotted (across and down, one panel each).

usage:
  slipcam-regions run/offsets.npz --region driveway 150 450 1200 650 --region field 980 140 1200 220 \
      --out run/series
needs: numpy, matplotlib
"""

import argparse

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt     # noqa: E402

# the categorical slots of the reference palette, in order
SERIES = ['#2a78d6', '#eb6834', '#1baf7a', '#eda100', '#e87ba4', '#008300', '#6250d6', '#e34948']
INK, INK2, GRID = '#0b0b0b', '#52514e', '#e4e3df'


def region_series(z, box, min_snr):
    """The median offsets (across, down) of the windows of a box, per frame, and their numbers"""
    x0, y0, x1, y1 = box
    inside = ((z['yc'][:, None] >= y0) & (z['yc'][:, None] < y1)
              & (z['xc'][None, :] >= x0) & (z['xc'][None, :] < x1) & ~z['masked'])
    off = z['offsets'][:, inside]                                   # frames x windows x (down, across)
    ok = z['snr'][:, inside] >= min_snr
    off = np.where(ok[..., None], off, np.nan)
    with np.errstate(all='ignore'):
        med = np.nanmedian(off, axis=1)
    return med[:, 1], med[:, 0], ok.sum(axis=1), inside.sum()


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('offsets', help='the .npz of slipcam-track')
    parser.add_argument('--region', nargs=5, action='append', required=True, metavar=('NAME', 'X0', 'Y0', 'X1', 'Y1'),
                        help='a region, pixels (repeatable)')
    parser.add_argument('--min-snr', type=float, default=4, help='the smallest SNR of the windows used (default: %(default)s)')
    parser.add_argument('--fps', type=float, default=30, help='the frame rate (default: %(default)s)')
    parser.add_argument('--tmin', type=float, help='the start of the time axis, s')
    parser.add_argument('--tmax', type=float, help='the end of the time axis, s')
    parser.add_argument('--out', required=True, help='the output prefix (.csv and .png)')
    args = parser.parse_args()

    z = np.load(args.offsets)
    t = (z['index'] - int(z['ref'])) / args.fps
    columns, header = [z['index'], t], ['frame', 't_s']
    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, constrained_layout=True)
    for k, (name, *box) in enumerate(args.region):
        across, down, nok, nwin = region_series(z, [float(v) for v in box], args.min_snr)
        print(f'{name:12s} {nwin:4d} windows, {np.median(nok):.0f} used per frame (median); '
              f'final across {np.nanmedian(across[-30:]):+.2f}, down {np.nanmedian(down[-30:]):+.2f} px')
        columns += [across, down, nok]
        header += [f'{name}_across_px', f'{name}_down_px', f'{name}_nwin']
        for ax, series in zip(axes, (across, down)):
            ax.plot(t, series, color=SERIES[k], lw=1.6, label=name)
    np.savetxt(args.out + '.csv', np.column_stack(columns), delimiter=',', header=','.join(header), comments='',
               fmt='%.4f')

    for ax, name in zip(axes, ('Across (+ right), pixels', 'Down (+ down), pixels')):
        ax.set_ylabel(name, color=INK2, fontsize=9)
        ax.axhline(0, color=INK2, lw=0.8)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=INK2, labelsize=8)
        for s in ax.spines.values():
            s.set_visible(False)
    axes[0].legend(frameon=False, fontsize=9, ncol=len(args.region), loc='upper left', labelcolor=INK)
    axes[0].set_title('Median offset of each region relative to the reference frame', color=INK, fontsize=11, loc='left')
    axes[1].set_xlabel('Time after the reference frame, s', color=INK2, fontsize=9)
    axes[1].set_xlim(args.tmin if args.tmin is not None else t[0], args.tmax if args.tmax is not None else t[-1])
    fig.savefig(args.out + '.png', dpi=110, facecolor='#fcfcfb')
    print(f'-> {args.out}.csv, {args.out}.png')


if __name__ == '__main__':
    main()
