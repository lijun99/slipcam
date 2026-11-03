#!/usr/bin/env python3
"""
The fault slip seen by the camera: the offsets of the far side relative to the camera side

The camera moves during the shaking (shifts, tilts and rolls), which moves the whole image. Per frame,
a similarity transform (shift p, q, roll a and scale s about the image center),

    across = p - a (y - yc) + s (x - xc),    down = q + a (x - xc) + s (y - yc),

is fitted (robust least squares) to the windows of the camera side of the fault (--near, boxes of
the frame), and removed from all the windows; the residual offsets are the motion of the ground
relative to the camera side. The residuals of the far side (--far) are its slip, in pixels; with a
uniform scale (--scale, m per pixel) or a final slip to calibrate it (--final-slip, m), in meters; with
scale bars (--scale-bar) or ground control points (--control), each window converted to the ground at
its own scale (slipcam.scale), the slip along its final direction, in meters.

Outputs: the residual offsets (.npz), a map of the final residuals, and the time series of the far
side (.csv, .png): the slip (across, the fault-parallel direction in the image), its rate, and the
fit of the camera motion.

usage:
  slipcam-slip run/offsets.npz run/frames.npy --near 150 450 1200 650 --far 980 130 1220 230 \
      --final 770 806 --final-slip 2.5 --out run/slip
needs: numpy, matplotlib
"""

import argparse
import warnings

import matplotlib
import numpy as np

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402

from .offset_map import DIVERGING, INK, INK2        # noqa: E402
from .region_series import SERIES, GRID              # noqa: E402
from . import scale as scales                       # noqa: E402


def in_boxes(z, boxes):
    """The windows (centers) inside any of the boxes, and not masked"""
    sel = np.zeros(z['masked'].shape, bool)
    for x0, y0, x1, y1 in boxes:
        sel |= ((z['yc'][:, None] >= y0) & (z['yc'][:, None] < y1) & (z['xc'][None, :] >= x0) & (z['xc'][None, :] < x1))
    return sel & ~z['masked']


def fit_camera(dx, dy, x, y, model='similarity', iterations=3):
    """A robust fit of the similarity transform (p, q, a, s) to the offsets dx (across), dy (down);
    with model 'shift', of the shift (p, q) only, a and s being 0"""
    ok = np.isfinite(dx) & np.isfinite(dy)
    n = len(x)
    G = np.zeros((2 * n, 4))
    G[:n] = np.column_stack([np.ones(n), np.zeros(n), -y, x])
    G[n:] = np.column_stack([np.zeros(n), np.ones(n), x, y])
    if model == 'shift':
        G[:, 2:] = 0
    d = np.concatenate([dx, dy])
    use = np.concatenate([ok, ok])
    m = np.full(4, np.nan)
    for _ in range(iterations):
        if use.sum() < 8:
            return np.full(4, np.nan), 0
        m = np.linalg.lstsq(G[use], d[use], rcond=None)[0]
        r = np.where(use, d - G @ m, np.nan)
        mad = 1.4826 * np.nanmedian(np.abs(r - np.nanmedian(r)))
        keep = np.abs(r) < max(3 * mad, 0.3)
        use = use & keep
        use = use[:n] & use[n:]
        use = np.concatenate([use, use])
    return m, use[:n].sum()


def pulse_metrics(t, series, final, smooth=5, hold=30):
    """The rise (10-90 %) of the main step of a slip series, and its peak rate

    The frames without a value (e.g., motion blur) are interpolated, not zeroed; the series is smoothed
    over smooth frames; the main step reaches 90 % of final and stays above 50 % for hold frames, and
    starts at the last frame below 10 % before it; the peak rate is the largest in the rise.
    """
    gap = ~np.isfinite(series)
    filled = np.interp(t, t[~gap], series[~gap])
    smoothed = np.convolve(filled, np.ones(smooth) / smooth, mode='same')
    rate = np.gradient(smoothed, t)
    held = np.array([smoothed[k] >= 0.9 * final and np.all(smoothed[k:k + hold] >= 0.5 * final)
                     for k in range(len(t))])
    k90 = np.argmax(held)
    below = np.flatnonzero(smoothed[:k90] < 0.1 * final)
    k10 = below[-1] if len(below) else 0
    kpeak = k10 + np.argmax(rate[k10:k90 + 1])
    return dict(gap=gap, smoothed=smoothed, rate=rate, k10=k10, k90=k90, kpeak=kpeak,
                t10=t[k10], t90=t[k90], peak_rate=rate[kpeak], t_peak=t[kpeak])


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('offsets', help='the .npz of slipcam-track')
    parser.add_argument('frames', help='the .npy of the frames (for the map)')
    parser.add_argument('--near', type=float, nargs=4, action='append', required=True, metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='a box of the camera side, pixels (repeatable)')
    parser.add_argument('--far', type=float, nargs=4, action='append', required=True, metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='a box of the far side, pixels (repeatable)')
    parser.add_argument('--model', choices=('similarity', 'shift'), default='similarity',
                        help='the camera motion: similarity, or shift only for --near and --far next to each other '
                             'in the image (default: %(default)s)')
    parser.add_argument('--min-snr', type=float, default=4, help='the smallest SNR of the windows used (default: %(default)s)')
    parser.add_argument('--fps', type=float, default=30, help='the frame rate (default: %(default)s)')
    parser.add_argument('--final', type=int, nargs=2, required=True, metavar=('FIRST', 'LAST'),
                        help='the frames of the final (static) offsets')
    parser.add_argument('--smooth', type=int, default=5, help='the smoothing of the slip for its rate, frames (default: %(default)s)')
    parser.add_argument('--hold', type=int, default=30, help='the frames the slip stays above 50 %% after its rise (default: %(default)s)')
    parser.add_argument('--scale', type=float, help='meters per pixel of the far side')
    parser.add_argument('--final-slip', type=float, help='the final slip, m, to calibrate the scale')
    scales.add_arguments(parser)
    parser.add_argument('--tmin', type=float, help='the start of the time axis, s')
    parser.add_argument('--tmax', type=float, help='the end of the time axis, s')
    parser.add_argument('--vmax', type=float, default=15, help='the limit of the color scale of the map, pixels')
    parser.add_argument('--out', required=True, help='the output prefix')
    args = parser.parse_args()
    model = scales.from_args(args)
    if model and (args.scale or args.final_slip):
        parser.error('--scale or --final-slip, or --scale-bar or --control, not both')

    z = np.load(args.offsets)
    off, snr = z['offsets'].copy(), z['snr']
    off[(snr < args.min_snr) | z['masked'][None]] = np.nan
    ref = np.load(args.frames, mmap_mode='r')[int(z['ref'])]
    h, w = ref.shape
    X, Y = np.meshgrid(z['xc'] - w / 2, z['yc'] - h / 2)
    near, far = in_boxes(z, args.near), in_boxes(z, args.far)
    t = (z['index'] - int(z['ref'])) / args.fps

    # the camera motion, per frame, and the residual offsets
    cam = np.full((len(t), 4), np.nan)
    nfit = np.zeros(len(t), int)
    res = np.full_like(off, np.nan)
    for k in range(len(t)):
        cam[k], nfit[k] = fit_camera(off[k][near][:, 1], off[k][near][:, 0], X[near], Y[near], args.model)
        p, q, a, s = cam[k]
        res[k, ..., 1] = off[k, ..., 1] - (p - a * Y + s * X)
        res[k, ..., 0] = off[k, ..., 0] - (q + a * X + s * Y)
    np.savez(args.out + '_residuals.npz', index=z['index'], t=t, residuals=res, camera=cam, nfit=nfit,
             near=near, far=far, xc=z['xc'], yc=z['yc'])

    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice')    # frames without valid windows
        across = np.nanmedian(res[:, far, 1], axis=1)
        down = np.nanmedian(res[:, far, 0], axis=1)
        near_across = np.nanmedian(res[:, near, 1], axis=1)
        nfar = np.isfinite(res[:, far, 1]).sum(axis=1)
        fin = (z['index'] >= args.final[0]) & (z['index'] <= args.final[1])
        final_map = np.nanmedian(res[fin], axis=0)
    final_px = np.nanmedian(across[fin])
    scale = args.scale or (args.final_slip / final_px if args.final_slip else None)

    print(f'camera fit: median {np.median(nfit):.0f} windows of {near.sum()}; final roll '
          f'{np.degrees(np.nanmedian(cam[fin, 2])):+.2f} deg, shift ({np.nanmedian(cam[fin, 0]):+.1f}, '
          f'{np.nanmedian(cam[fin, 1]):+.1f}) px')
    print(f'far side: {far.sum()} windows, median {np.median(nfar):.0f} used; final across {final_px:+.2f} px, '
          f'down {np.nanmedian(down[fin]):+.2f} px; camera side residual {np.nanmedian(near_across[fin]):+.2f} px')

    m = pulse_metrics(t, across, final_px, args.smooth, args.hold)
    gap, rate, t10, t90, kpeak = m['gap'], m['rate'], m['t10'], m['t90'], m['kpeak']
    print(f'{gap.sum()} frames without a slip (interpolated): {z["index"][gap].tolist()}')
    print(f'slip: 10 % at {t10:.2f} s, 90 % at {t90:.2f} s (rise {t90 - t10:.2f} s); peak rate {rate[kpeak]:.1f} px/s '
          f'at {t[kpeak]:.2f} s (smoothed over {args.smooth} frames)')
    if scale:
        print(f'scale {scale * 100:.2f} cm/px: final slip {final_px * scale:.2f} m, peak rate {rate[kpeak] * scale:.2f} m/s')

    # the slip on the ground, each window at its own scale
    if model:
        XC, YC = np.meshgrid(z['xc'], z['yc'])
        g = scales.slip(model, XC[far], YC[far], res[:, far][..., ::-1], fin)
        with np.errstate(all='ignore'), warnings.catch_warnings():
            warnings.filterwarnings('ignore', 'All-NaN slice')    # frames without valid windows
            near_m = np.nanmedian(model.ground(XC[near], YC[near], res[:, near][..., ::-1]) @ g['direction'], axis=1)
        final_m = float(np.nanmedian(g['along'][fin]))
        mm = pulse_metrics(t, g['along'], final_m, args.smooth, args.hold)
        local = model.along(np.mean(XC[far]), np.mean(YC[far]), [final_px, np.nanmedian(down[fin])])
        azimuth = f', azimuth {g["azimuth"]:.0f} deg' if 'azimuth' in g else ''
        print(f'{model.info}; {local * 100:.2f} cm/px along the slip at the far side')
        print(f'slip on the ground: final {final_m:.2f} m{azimuth}, normal to it {np.nanmedian(g["normal"][fin]):+.2f} m; '
              f'rise {mm["t90"] - mm["t10"]:.2f} s; peak rate {mm["peak_rate"]:.2f} m/s at {mm["t_peak"]:.2f} s')

    cols = [z['index'], t, across, down, near_across, nfar, cam[:, 0], cam[:, 1], np.degrees(cam[:, 2]), cam[:, 3]]
    head = 'frame,t_s,far_across_px,far_down_px,near_across_px,far_nwin,cam_p_px,cam_q_px,cam_roll_deg,cam_scale'
    if model:
        cols += [g['along'], g['normal'], near_m]
        head += ',far_slip_m,far_normal_m,near_slip_m'
    np.savetxt(args.out + '.csv', np.column_stack(cols), delimiter=',', header=head, comments='', fmt='%.5f')

    # the map of the final residuals
    skip, xc, yc = int(z['skip']), z['xc'], z['yc']
    extent = (xc[0] - skip / 2, xc[-1] + skip / 2, yc[-1] + skip / 2, yc[0] - skip / 2)
    fig, axes = plt.subplots(1, 2, figsize=(15, 4.9), constrained_layout=True)
    for ax, comp, name in zip(axes, (1, 0), ('across (+ right)', 'down (+ down)')):
        ax.imshow(ref, cmap='gray', vmin=0, vmax=255, extent=(0, w, h, 0))
        im = ax.imshow(final_map[..., comp], cmap=DIVERGING, vmin=-args.vmax, vmax=args.vmax, alpha=0.75,
                       extent=extent, interpolation='nearest')
        for boxes, color, label in ((args.near, SERIES[0], 'camera side'), (args.far, SERIES[1], 'far side')):
            for k, (x0, y0, x1, y1) in enumerate(boxes):
                ax.add_patch(plt.Rectangle((x0, y0), x1 - x0, y1 - y0, fill=False, ec=color, lw=1.6,
                                           label=label if k == 0 else None))
        ax.set_xlim(0, w)
        ax.set_ylim(h, 0)
        ax.set_title(f'Final offset {name} after removing the camera motion', color=INK, fontsize=11, loc='left')
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
        cb = fig.colorbar(im, ax=ax, shrink=0.85, pad=0.01)
        cb.set_label('pixels', color=INK2, fontsize=9)
        cb.ax.tick_params(colors=INK2, labelsize=8)
        cb.outline.set_visible(False)
    axes[0].legend(loc='lower right', fontsize=8, frameon=True, framealpha=0.9, labelcolor=INK)
    fig.savefig(args.out + '_map.png', dpi=110, facecolor='#fcfcfb')

    # the time series: slip and its rate (one panel each, one axis each), and the camera roll
    if model:
        unit, lines = 'm', (g['along'], g['normal'], near_m)
        names = ('far side, along its final slip', 'far side, normal to it', 'camera side, along it (should stay at 0)')
        rate, kpeak, t10, t90 = mm['rate'], mm['kpeak'], mm['t10'], mm['t90']
    else:
        f = scale or 1.0
        unit, lines = 'm' if scale else 'pixels', (across * f, down * f, near_across * f)
        names = ('far side, across (fault-parallel)', 'far side, down', 'camera side, across (should stay at 0)')
        rate = rate * f
    roll = args.model == 'similarity'
    fig, axes = plt.subplots(3 if roll else 2, 1, figsize=(11, 8 if roll else 6.5), sharex=True,
                             constrained_layout=True, gridspec_kw=dict(height_ratios=(3, 2, 1.5)[:3 if roll else 2]))
    for series, color, lw, name in zip(lines, (SERIES[1], SERIES[2], SERIES[0]), (1.8, 1.4, 1.4), names):
        axes[0].plot(t, series, color=color, lw=lw, label=name)
    axes[0].axvspan(t10, t90, color=GRID, alpha=0.6, lw=0)
    axes[0].set_ylabel(f'Slip, {unit}', color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK)
    title = 'Fault slip from the CCTV video, relative to the camera side'
    if scale and args.final_slip:
        title += f' (scale calibrated to a final slip of {args.final_slip} m)'
    elif model:
        title += f' ({model.info})'
    axes[0].set_title(title, color=INK, fontsize=11, loc='left')
    for tg in t[gap]:
        axes[0].axvline(tg, color=INK2, lw=0.6, ls=':')
    axes[1].plot(t, rate, color=SERIES[1], lw=1.6)
    axes[1].set_ylabel(f'Slip rate, {unit}/s ({args.smooth}-frame smoothing)', color=INK2, fontsize=9)
    axes[1].annotate(f'peak {rate[kpeak]:.{2 if unit == "m" else 1}f} {unit}/s', (t[kpeak], rate[kpeak]), xytext=(8, -4),
                     textcoords='offset points', color=INK, fontsize=9)
    if roll:
        axes[2].plot(t, np.degrees(cam[:, 2]), color=SERIES[0], lw=1.4)
        axes[2].set_ylabel('Camera roll, deg', color=INK2, fontsize=9)
    axes[-1].set_xlabel('Time after the reference frame (clock 12:46:30), s', color=INK2, fontsize=9)
    for ax in axes:
        ax.axhline(0, color=INK2, lw=0.8)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    tmin = args.tmin if args.tmin is not None else t[0]
    tmax = args.tmax if args.tmax is not None else t[-1]
    axes[-1].set_xlim(tmin, tmax)
    shown = (t >= tmin) & (t <= tmax)
    lo, hi = np.min(rate[shown]), np.max(rate[shown])
    axes[1].set_ylim(min(lo, 0) - 0.1 * (hi - lo), hi + 0.15 * (hi - lo))
    fig.savefig(args.out + '.png', dpi=110, facecolor='#fcfcfb')
    print(f'-> {args.out}.csv, {args.out}.png, {args.out}_map.png, {args.out}_residuals.npz')


if __name__ == '__main__':
    main()
