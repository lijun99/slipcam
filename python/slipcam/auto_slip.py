#!/usr/bin/env python3
"""
Fault slip from a video, without choosing regions by hand

From the frames of slipcam-decode:

1. timing: the frozen frames (identical to the previous one), the onset of the shaking (the jump of the
   differences of consecutive frames), and the reference frame, the last calm frame before it; if there
   are no calm frames (a clip starting during the shaking, a phone filming a screen), the first frame,
   and the frames are first stabilized on it (slipcam.stabilize, --stabilize);
2. text: the burned-in text, pixels of strong edges that do not change while the scene shakes; the
   windows over it are masked;
3. tracking: the offsets of the frames relative to the reference frame, with framecor (or pycuampcor,
   --engine) on the GPU (track_offsets.track);
4. quality: the windows of a periodic or flat texture, whose correlation with their own surroundings in
   the reference frame has a second peak (--max-periodicity), are flagged, and the offsets of low SNR
   (--min-snr) dropped;
5. sides: the final offsets relative to a robust fit of the camera motion (shift, roll, scale) to all
   the windows, most of them on the camera side; the far side has residuals above 4 times their robust
   spread, consistent with its neighbors and stable at the end; the camera side, below half of that;
6. camera motion: each far-side window is paired with the camera-side windows near it in the image
   whose motion before the slip matches its own best (--pairs); their median is removed from its
   offsets, which cancels the motion of the camera without a model of it;
7. slip: per region of the far side (neighbors of similar final offsets), the median of the paired
   differences of its windows,
   its spread, and the rise and peak rate of the slip (slip_series.pulse_metrics), in pixels.

usage:
  slipcam-auto ../data/run/frames.npy --out ../data/auto/myanmar
  slipcam-auto ../data/run/frames.npy --out ../data/auto/myanmar --explain 980 130 1220 200   # why a box was used
needs: numpy, scipy, matplotlib, framecor (or pycuampcor)
"""

import argparse
import json
import os

import matplotlib
import numpy as np
from scipy import ndimage, signal

matplotlib.use('Agg')
import matplotlib.pyplot as plt                     # noqa: E402

from .offset_map import INK, INK2                    # noqa: E402
from .region_series import SERIES, GRID              # noqa: E402
from .slip_series import fit_camera, pulse_metrics   # noqa: E402
from .stabilize import coarse_track, stabilize       # noqa: E402
from .track_offsets import track, window_grid        # noqa: E402


def frame_differences(frames, chunk=50):
    """The mean absolute difference of consecutive frames"""
    d = np.empty(len(frames) - 1, np.float32)
    for a in range(0, len(frames) - 1, chunk):
        b = min(a + chunk + 1, len(frames))
        f = frames[a:b].astype(np.float32)
        d[a:b - 1] = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    return d


def timing(d, frozen_level=0.05, lead=5, calm=30):
    """The frozen frames, the first shaking frame and the reference frame, from the frame differences;
    the level of the shaking is 10 times that of the first calm frames after a leading freeze"""
    frozen = np.concatenate([[False], d < frozen_level])        # frame k identical to frame k-1
    first_live = np.argmax(~frozen[1:]) + 1 if frozen[1] else 0
    thr = max(0.5, 10 * np.median(d[first_live:first_live + calm]))
    run = np.array([np.median(d[k:k + 5]) for k in range(len(d))])
    above = run[first_live:] > thr
    if not above.any():                                          # no jump: shaking from the start, or none
        return frozen, None, first_live, thr
    onset = first_live + np.argmax(above) + 1                    # the first frame that moves
    ref = max(onset - lead, first_live)
    return frozen, onset, ref, thr


def text_mask(frames, start, nframes=150, chunk=10, edge_percentile=90, max_std=1.5, dilate=4):
    """The pixels of burned-in text: strong edges whose values do not change while the scene shakes

    The temporal standard deviation is the median over chunks of frames, so that text that changes once
    per chunk at most (e.g., the seconds of a clock) is still found."""
    stds = []
    for a in range(start, min(start + nframes, len(frames)) - chunk + 1, chunk):
        stds.append(frames[a:a + chunk].astype(np.float32).std(axis=0))
    std = np.median(stds, axis=0)
    ref = frames[start].astype(np.float32)
    edge = np.hypot(ndimage.sobel(ref, 0), ndimage.sobel(ref, 1))
    text = (edge > np.percentile(edge, edge_percentile)) & (std < max_std)
    # the flat inside of caption boxes is not an edge: fill the regions enclosed by the text
    return ndimage.binary_fill_holes(ndimage.binary_dilation(text, iterations=dilate))


def window_fraction(mask, y0, x0, window):
    """The fraction of the pixels of each window in a mask"""
    cs = np.pad(mask.astype(np.float64).cumsum(0).cumsum(1), ((1, 0), (1, 0)))
    y1, x1 = y0 + window, x0 + window
    s = cs[y1][:, x1] - cs[y1][:, x0] - cs[y0][:, x1] + cs[y0][:, x0]
    return s / window ** 2


def periodicity(ref, y0, x0, window, search, exclude=5):
    """The second peak of the normalized correlation of each window with its own surroundings in the
    reference frame (1 for a periodic or flat texture, small for a unique one)"""
    ref = ref.astype(np.float64)
    out = np.ones((len(y0), len(x0)))
    ones = np.ones((window, window))
    n = window * window
    for i, y in enumerate(y0):
        for j, x in enumerate(x0):
            t = ref[y:y + window, x:x + window]
            s = ref[y - search:y + window + search, x - search:x + window + search]
            t0 = t - t.mean()
            tn = np.sqrt((t0 ** 2).sum())
            if tn < 1e-6:
                continue
            num = signal.fftconvolve(s, t0[::-1, ::-1], mode='valid')
            ssum = signal.fftconvolve(s, ones, mode='valid')
            ssq = signal.fftconvolve(s ** 2, ones, mode='valid')
            var = np.maximum(ssq - ssum ** 2 / n, 1e-6)
            ncc = num / (np.sqrt(var) * tn)
            c = search
            peaks = (ncc == ndimage.maximum_filter(ncc, size=5))
            peaks[c - exclude:c + exclude + 1, c - exclude:c + exclude + 1] = False
            out[i, j] = ncc[peaks].max() if peaks.any() else 0
    return out


def split_sides(off, fin, usable, X, Y, max_final_std, min_valid=0.3):
    """The camera side and the far side of the fault, from the final offsets (the median over the frames
    fin) relative to a robust similarity fit of the camera motion (X, Y: the window centers relative to
    the image center): the far side has residuals above 4 times their robust spread, consistent with the
    median of its neighbors and stable over fin (robust scatter below max_final_std); the camera side
    is the rest"""
    with np.errstate(all='ignore'):
        final = np.nanmedian(off[fin], axis=0)[..., ::-1]               # (across, down)
        nvalid = np.isfinite(off[fin][..., 0]).mean(axis=0)
        mad = 1.4826 * np.nanmedian(np.abs(off[fin] - np.nanmedian(off[fin], axis=0)), axis=0)
    good = usable & (nvalid >= min_valid) & np.isfinite(final).all(-1)
    cam, _ = fit_camera(final[good][:, 0], final[good][:, 1], X[good], Y[good])
    p, q, a, s = cam
    resid = np.stack([final[..., 0] - (p - a * Y + s * X), final[..., 1] - (q + a * X + s * Y)], axis=-1)
    rmag = np.where(good, np.hypot(resid[..., 0], resid[..., 1]), np.nan)
    sigma = 1.4826 * np.nanmedian(np.abs(rmag - np.nanmedian(rmag)))
    thr = max(4 * sigma, 1.0)
    with np.errstate(all='ignore'):
        nbr = np.stack([ndimage.generic_filter(np.where(good, resid[..., c], np.nan), np.nanmedian, size=3,
                                               mode='constant', cval=np.nan) for c in (0, 1)], axis=-1)
        unstable = np.nanmax(mad, axis=-1) > max_final_std
    consistent = np.hypot(*(resid - nbr).transpose(2, 0, 1)) < thr
    big = good & (rmag > thr)
    return dict(final=final, mad=mad, good=good, cam=cam, resid=resid, rmag=rmag, sigma=sigma, thr=thr,
                consistent=consistent, unstable=unstable, big=big, far=big & consistent & ~unstable, near=good & ~big)


def slip_onset(off, t, fin, sides, smooth):
    """The rise of the slip from the difference of the medians of the two sides, along their separation"""
    far, near = sides['far'], sides['near']
    with np.errstate(all='ignore'):
        sep = np.nanmedian(sides['final'][far], axis=0) - np.nanmedian(sides['final'][near], axis=0)
        u = sep / np.hypot(*sep)
        gdiff = (np.nanmedian(off[:, far][..., ::-1], axis=1) - np.nanmedian(off[:, near][..., ::-1], axis=1)) @ u
    return pulse_metrics(t, gdiff, np.nanmedian(gdiff[fin]), smooth)


def hampel(x, half=3, nsig=3, floor=1.0):
    """The series x (frames x ...) with its outliers (beyond nsig robust deviations, at least floor, from
    the running median over 2 half + 1 frames) set to NaN; ramps are kept, single-frame dropouts removed"""
    pad = np.pad(x, [(half, half)] + [(0, 0)] * (x.ndim - 1), mode='constant', constant_values=np.nan)
    win = np.lib.stride_tricks.sliding_window_view(pad, 2 * half + 1, axis=0)
    with np.errstate(all='ignore'):
        med = np.nanmedian(win, axis=-1)
        mad = 1.4826 * np.nanmedian(np.abs(win - med[..., None]), axis=-1)
    return np.where(np.abs(x - med) > np.maximum(nsig * mad, floor), np.nan, x)


def similar_regions(sel, values, tol):
    """The regions of the selected windows, connected through neighbors (8 of them) whose values
    (... x 2) differ by less than tol; labels from 1, and their number"""
    labels = np.zeros(sel.shape, int)
    ny, nx = sel.shape
    count = 0
    for i0, j0 in zip(*np.nonzero(sel)):
        if labels[i0, j0]:
            continue
        count += 1
        labels[i0, j0] = count
        stack = [(i0, j0)]
        while stack:
            i, j = stack.pop()
            for di in (-1, 0, 1):
                for dj in (-1, 0, 1):
                    a, b = i + di, j + dj
                    if (0 <= a < ny and 0 <= b < nx and sel[a, b] and not labels[a, b]
                            and np.hypot(*(values[a, b] - values[i, j])) < tol):
                        labels[a, b] = count
                        stack.append((a, b))
    return labels, count


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('frames', help='the .npy of the frames (frames x height x width, uint8)')
    parser.add_argument('--fps', type=float, default=30, help='the frame rate (default: %(default)s)')
    parser.add_argument('--window', type=int, default=64, help='the window size, pixels (default: %(default)s)')
    parser.add_argument('--skip', type=int, default=16, help='the distance between windows, pixels (default: %(default)s)')
    parser.add_argument('--search', type=int, default=48, help='the half search range, pixels (default: %(default)s)')
    parser.add_argument('--min-snr', type=float, default=4, help='the smallest SNR of the offsets used (default: %(default)s)')
    parser.add_argument('--max-periodicity', type=float, default=0.8,
                        help='the largest second correlation peak of the windows used (default: %(default)s)')
    parser.add_argument('--final-seconds', type=float, default=4,
                        help='the end of the video taken as the final offsets, s (default: %(default)s)')
    parser.add_argument('--min-valid', type=float, default=0.3,
                        help='the smallest fraction of valid offsets of a window over the final frames (default: %(default)s)')
    parser.add_argument('--settle', type=float, default=1,
                        help='the time after the slip before the final offsets, s (default: %(default)s)')
    parser.add_argument('--max-final-std', type=float, default=4,
                        help='the largest robust scatter (MAD) of the offsets of a window at the end, pixels (default: %(default)s)')
    parser.add_argument('--pairs', type=int, default=5, help='the camera-side windows paired with each far-side one (default: %(default)s)')
    parser.add_argument('--radius', type=float, default=200, help='the search radius of the pairs in the image, pixels (default: %(default)s)')
    parser.add_argument('--min-region', type=int, default=6, help='the smallest far-side region, windows (default: %(default)s)')
    parser.add_argument('--min-score', type=float, default=5,
                        help='the smallest score of a region that looks like slip (default: %(default)s)')
    parser.add_argument('--smooth', type=int, default=5, help='the smoothing of the slip for its rate, frames (default: %(default)s)')
    parser.add_argument('--stabilize', choices=('auto', 'on', 'off'), default='auto',
                        help='warp the frames onto the reference first: auto, if there are no calm frames before '
                             'the shaking (default: %(default)s)')
    parser.add_argument('--engine', choices=('framecor', 'pycuampcor'), default='framecor',
                        help='the correlation engine (default: %(default)s)')
    parser.add_argument('--cpu', action='store_true', help='correlate on the CPU')
    parser.add_argument('--explain', type=float, nargs=4, action='append', default=[], metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='print why the windows of a box were used or not (repeatable)')
    parser.add_argument('--out', required=True, help='the output prefix')
    args = parser.parse_args()

    os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
    frames = np.load(args.frames, mmap_mode='r')
    n, h, w = frames.shape
    summary = {}

    # 1. timing
    d = frame_differences(frames)
    frozen, onset, ref, thr = timing(d)
    warnings = []
    if onset is None or onset - ref < 5:
        warnings.append('no calm frames before the shaking (the clip may start during it, or the camera moves '
                        'all the time, e.g., a phone filming a screen): the reference frame is the first one')
    repeated = int((d[onset or ref:] < 0.05).sum())
    print(f'1. timing: {frozen.sum()} frozen frames, first at {np.flatnonzero(frozen)[:1].tolist()}; shaking from '
          f'frame {onset} (threshold {thr:.2f}); reference frame {ref}; {repeated} repeated frames after the onset')
    for wmsg in warnings:
        print(f'   WARNING: {wmsg}')
    summary['timing'] = dict(frozen=int(frozen.sum()), onset=None if onset is None else int(onset), ref=int(ref),
                             repeated_after_onset=repeated)
    summary['warnings'] = warnings

    # 2. text
    _, y0, x0 = window_grid(h, w, args.window, args.skip, args.search)
    text = text_mask(frames, onset or ref)
    print(f'2. text: {text.mean() * 100:.2f} % of the pixels')

    # stabilization, if the camera moves all the time (no calm frames before the shaking)
    stab = args.stabilize == 'on' or (args.stabilize == 'auto' and bool(warnings))
    if stab:
        params, used, ngrid = coarse_track(frames, ref, text)
        bad = int((used[ref:] < 6).sum())
        print(f'   stabilization: {ngrid} coarse windows; median {np.median(used[ref:]):.0f} used per frame, '
              f'{bad} frames with fewer than 6; largest shift {np.nanmax(np.hypot(params[:, 0], params[:, 1])):.0f} px, '
              f'rotation {np.degrees(np.nanmax(np.abs(params[:, 2]))):.1f} deg, scale {np.nanmax(np.abs(params[:, 3])) * 100:.1f} %')
        stabilized, _ = stabilize(frames, params, text)   # the text is blanked; the windows over it in the reference are masked
        np.save(args.out + '_stabilized.npy', stabilized)
        np.savez(args.out + '_stabilization.npz', params=params, used=used)
        frames = np.load(args.out + '_stabilized.npy', mmap_mode='r')
        d2 = frame_differences(frames)
        _, onset2, ref2, thr2 = timing(d2)
        if onset2 is not None and onset2 - ref2 >= 5:
            onset, ref = onset2, ref2
            print(f'   after stabilization: shaking from frame {onset} (threshold {thr2:.2f}); reference frame {ref}')
        summary['stabilization'] = dict(coarse_windows=int(ngrid), frames_with_few_windows=bad, ref=int(ref))
    frac = window_fraction(text, y0, x0, args.window)
    masked = frac > 0.02
    print(f'   {masked.sum()} windows masked of {masked.size}')
    summary['text'] = dict(pixels_percent=float(text.mean() * 100), windows_masked=int(masked.sum()))

    # 3. tracking (cached)
    cache = args.out + '_offsets.npz'
    if (os.path.exists(cache) and int(np.load(cache)['ref']) == ref and np.array_equal(np.load(cache)['masked'], masked)
            and bool(np.load(cache).get('stabilized', False)) == stab
            and str(np.load(cache).get('engine', 'pycuampcor')) == args.engine):
        z = dict(np.load(cache))
        print(f'3. tracking: cached in {cache}')
    else:
        z = track(frames, ref, np.arange(ref, n), args.out + '_ampcor', masked=masked,
                  window=args.window, skip=args.skip, search=args.search, cpu=args.cpu, engine=args.engine)
        z['stabilized'] = stab
        np.savez(cache, **z)
    off, snr = z['offsets'], z['snr']
    t = (z['index'] - ref) / args.fps

    # 4. quality
    per = periodicity(frames[ref], y0, x0, args.window, args.search)
    periodic = per > args.max_periodicity
    usable = ~masked & ~periodic
    off = np.where(((snr >= args.min_snr) & usable[None])[..., None], off, np.nan)
    print(f'4. quality: {periodic.sum()} windows of periodic or flat texture; {usable.sum()} windows usable')
    summary['quality'] = dict(periodic=int(periodic.sum()), usable=int(usable.sum()))

    # 5. sides, from the final offsets relative to a robust fit of the camera motion; in two passes, the
    # second with the final offsets just after the slip (before, e.g., dust at the end of the video)
    X, Y = np.meshgrid(z['xc'] - w / 2, z['yc'] - h / 2)
    fin = t >= t[-1] - args.final_seconds
    S = split_sides(off, fin, usable, X, Y, args.max_final_std, args.min_valid)
    if not S['far'].any():
        msg = (f'no far side found (residual threshold {S["thr"]:.1f} px, {S["big"].sum()} windows above it, '
               f'{(S["big"] & ~S["consistent"]).sum()} inconsistent, {(S["big"] & S["unstable"]).sum()} unstable)')
        print(f'5. sides: {msg}; stopping')
        summary['warnings'].append(msg)
        with open(args.out + '_summary.json', 'w') as f:
            json.dump(summary, f, indent=2)
        return
    gm = slip_onset(off, t, fin, S, args.smooth)
    after = (t >= gm['t90'] + args.settle) & (t < gm['t90'] + args.settle + args.final_seconds)
    if after.sum() >= 30:
        fin = after
        S = split_sides(off, fin, usable, X, Y, args.max_final_std, args.min_valid)
        gm = slip_onset(off, t, fin, S, args.smooth)
    final, good, resid, rmag, mad = S['final'], S['good'], S['resid'], S['rmag'], S['mad']
    thr_side, consistent, unstable, far, near = S['thr'], S['consistent'], S['unstable'], S['far'], S['near']
    p, q, a, s = S['cam']
    Yc = np.broadcast_to(z['yc'][:, None], good.shape)
    bottom = good & (Yc > 2 * h / 3)
    print(f'5. sides: final offsets {t[fin][0]:.2f}-{t[fin][-1]:.2f} s; camera fit (shift {p:+.1f}, {q:+.1f} px, roll '
          f'{np.degrees(a):+.2f} deg); residual spread {S["sigma"]:.2f} px, threshold {thr_side:.1f} px; {near.sum()} '
          f'camera-side and {far.sum()} far-side windows ({(S["big"] & ~consistent).sum()} inconsistent, '
          f'{(S["big"] & consistent & unstable).sum()} unstable dropped); the camera side has {(near & bottom).sum()} '
          f'of the {bottom.sum()} windows of the bottom third')
    summary['sides'] = dict(camera_side=int(near.sum()), far_side=int(far.sum()), threshold_px=float(thr_side))
    for bx0, by0, bx1, by1 in args.explain:
        inb = ((Yc >= by0) & (Yc < by1) & (np.broadcast_to(z['xc'][None, :], good.shape) >= bx0)
               & (np.broadcast_to(z['xc'][None, :], good.shape) < bx1))
        big = good & (rmag > thr_side)
        print(f'   explain x {bx0:g}-{bx1:g}, y {by0:g}-{by1:g}: {inb.sum()} windows: {(inb & masked).sum()} text, '
              f'{(inb & periodic & ~masked).sum()} periodic or flat (second peak median {np.median(per[inb]):.2f}), '
              f'{(inb & usable & ~good).sum()} too few valid offsets at the end, {(inb & near).sum()} camera side, '
              f'{(inb & good & ~near & ~big).sum()} between the sides, {(inb & big & ~consistent).sum()} inconsistent, '
              f'{(inb & big & consistent & unstable).sum()} unstable (MAD median {np.nanmedian(np.nanmax(mad, -1)[inb]):.1f} px), '
              f'{(inb & far).sum()} far side')

    pre = (t > 0) & (np.arange(len(t)) < gm['k10'] - 6)
    print(f'   slip onset from the two sides: {gm["t10"]:.2f} s; {pre.sum()} frames of shaking before it for the pairing')

    # 6. camera motion: pairs by the motion before the slip; the camera-side windows within a window of the
    # far side (or of any large residual) are not partners, as their templates overlap it
    cells = int(np.ceil(args.window / args.skip))
    border = ndimage.binary_dilation(S['big'], structure=np.ones((3, 3)), iterations=cells)
    partner_ok = near & ~border
    iy, ix = np.nonzero(far)
    ny_, nx_ = np.nonzero(partner_ok)
    print(f'   partners: {partner_ok.sum()} camera-side windows, {(near & border).sum()} next to the far side excluded')
    pos_far = np.column_stack([z['yc'][iy], z['xc'][ix]])
    pos_near = np.column_stack([z['yc'][ny_], z['xc'][nx_]])
    near_off = off[:, partner_ok]                               # frames x partners x 2
    near_pre = near_off[pre]
    diffs = np.full((len(t), len(iy), 2), np.nan, np.float32)
    partners = np.zeros((len(iy), args.pairs), int)
    match = np.full(len(iy), np.nan)
    for k, (a, b) in enumerate(zip(iy, ix)):
        dist = np.hypot(*(pos_near - pos_far[k]).T)
        radius = args.radius
        cand = np.flatnonzero(dist < radius)
        while len(cand) < 2 * args.pairs and radius < max(h, w):
            radius *= 1.5
            cand = np.flatnonzero(dist < radius)
        with np.errstate(all='ignore'):
            rms = np.sqrt(np.nanmean(((near_pre[:, cand] - off[pre, a, b][:, None]) ** 2).sum(-1), axis=0))
        rms = np.where(np.isfinite(rms), rms, np.inf)
        best = cand[np.argsort(rms)[:args.pairs]]
        partners[k] = best
        match[k] = np.sort(rms)[:args.pairs].mean()
        with np.errstate(all='ignore'):
            diffs[:, k] = off[:, a, b] - np.nanmedian(near_off[:, best], axis=1)
    # single-frame dropouts (motion blur, dust) of each window
    before = np.isfinite(diffs[..., 0]).sum()
    diffs = hampel(diffs)
    print(f'6. pairing: {len(iy)} far-side windows, {args.pairs} camera-side windows each; median mismatch before '
          f'the slip {np.nanmedian(match):.2f} px; {before - np.isfinite(diffs[..., 0]).sum()} single-frame outliers removed')

    # 7. slip, per connected region of the far side
    regions, nreg = similar_regions(far, resid, thr_side)
    sizes = ndimage.sum(far, regions, range(1, nreg + 1))
    order = np.argsort(sizes)[::-1]
    keep = [int(r) + 1 for r in order if sizes[r] >= args.min_region]
    results = []
    for r in keep:
        sel = regions[iy, ix] == r
        dr = diffs[:, sel][..., ::-1]                            # frames x windows x (across, down)
        with np.errstate(all='ignore'):
            fw = np.nanmedian(dr[fin], axis=0)                   # final of each window
            fr = np.nanmedian(fw, axis=0)
            ur = fr / np.hypot(*fr)
            series = np.nanmedian(dr @ ur, axis=1)
            across, down = np.nanmedian(dr[..., 0], axis=1), np.nanmedian(dr[..., 1], axis=1)
            spread = np.nanpercentile(fw @ ur, [16, 84])
            lo, hi = np.nanpercentile(dr @ ur, [16, 84], axis=1)
        total = float(np.nanmedian(series[fin]))
        m = pulse_metrics(t, series, total, args.smooth)
        # how much the series looks like slip: a step, quiet before it, stable after it, its windows in agreement
        with np.errstate(all='ignore'):
            noise_pre = float(np.nanstd(series[pre]))
            noise_post = float(1.4826 * np.nanmedian(np.abs(series[fin] - total)))
        score = abs(total) / (spread[1] - spread[0] + noise_pre + noise_post + 1e-3)
        rows, cols = np.nonzero(regions == r)
        box = [float(z['xc'][cols.min()]), float(z['yc'][rows.min()]), float(z['xc'][cols.max()]), float(z['yc'][rows.max()])]
        res = dict(region=r, windows=int(sel.sum()), box_xy=box, score=float(score), final_across_px=float(fr[0]),
                   final_down_px=float(fr[1]), final_total_px=total, final_spread_px=spread.tolist(),
                   noise_before_px=noise_pre, noise_after_px=noise_post, t10_s=float(m['t10']), t90_s=float(m['t90']),
                   rise_s=float(m['t90'] - m['t10']), peak_rate_px_s=float(m['peak_rate']), t_peak_s=float(m['t_peak']),
                   onset_frame=int(z['index'][m['k10']]))
        np.savetxt(f'{args.out}_region{r}.csv', np.column_stack([z['index'], t, series, across, down, lo, hi]),
                   delimiter=',', header='frame,t_s,slip_px,across_px,down_px,p16_px,p84_px', comments='', fmt='%.4f')
        res['_series'], res['_lo'], res['_hi'] = series, lo, hi
        results.append(res)
    # the primary region: of the regions that look like slip (score >= --min-score), the largest slip, i.e.,
    # the far-side target nearest the camera (windows mixing the two sides show less slip); then by score
    results.sort(key=lambda r: -r['score'])
    clean = [r for r in results if r['score'] >= args.min_score]
    if clean:
        primary = max(clean, key=lambda r: abs(r['final_total_px']))
        results.remove(primary)
        results.insert(0, primary)
    for k, r in enumerate(results):
        r['primary'] = bool(clean) and k == 0
    for res in results:
        b = res['box_xy']
        print(f'7. region {res["region"]}{" (primary)" if res["primary"] else ""} (score {res["score"]:.1f}): {res["windows"]} windows, box x {b[0]:.0f}-{b[2]:.0f}, '
              f'y {b[1]:.0f}-{b[3]:.0f}; final {res["final_total_px"]:.1f} px (across {res["final_across_px"]:+.1f}, '
              f'down {res["final_down_px"]:+.1f}; windows 16-84 % {res["final_spread_px"][0]:.1f}-{res["final_spread_px"][1]:.1f}); '
              f'rise {res["t10_s"]:.2f}-{res["t90_s"]:.2f} s; peak rate {res["peak_rate_px_s"]:.1f} px/s at {res["t_peak_s"]:.2f} s')
    np.savez(args.out + '_pairs.npz', t=t, index=z['index'], far_yx=np.column_stack([iy, ix]),
             near_yx=np.column_stack([ny_, nx_]), partners=partners, diffs=diffs, regions=regions[iy, ix])
    summary['regions'] = [{k: v for k, v in r.items() if not k.startswith('_')} for r in results]
    summary['primary_region'] = results[0]['region'] if results and results[0]['primary'] else None
    with open(args.out + '_summary.json', 'w') as f:
        json.dump(summary, f, indent=2)

    # figures: the map of the automatic choices, and the slip of the primary region and the next one
    colors = {r['region']: SERIES[1 + k] for k, r in enumerate(results[:2])}
    img = np.zeros(good.shape + (4,))
    img[near] = matplotlib.colors.to_rgba(SERIES[0], 0.55)
    for r in keep:
        img[(regions == r)] = matplotlib.colors.to_rgba(colors.get(r, INK2), 0.6 if r in colors else 0.35)
    img[masked] = matplotlib.colors.to_rgba(INK, 0.5)
    skip = args.skip
    extent = (z['xc'][0] - skip / 2, z['xc'][-1] + skip / 2, z['yc'][-1] + skip / 2, z['yc'][0] - skip / 2)
    fig, ax = plt.subplots(figsize=(9, 5.4), constrained_layout=True)
    ax.imshow(frames[ref], cmap='gray', vmin=0, vmax=255, extent=(0, w, h, 0))
    ax.imshow(img, extent=extent, interpolation='nearest')
    handles = [plt.Rectangle((0, 0), 1, 1, color=SERIES[0], alpha=0.55, label='camera side')]
    handles += [plt.Rectangle((0, 0), 1, 1, color=colors[r['region']], alpha=0.6,
                              label=f'far side, region {r["region"]} ({r["final_total_px"]:.1f} px)') for r in results[:2]]
    if len(results) > 2:
        handles.append(plt.Rectangle((0, 0), 1, 1, color=INK2, alpha=0.35, label='far side, smaller regions'))
    handles.append(plt.Rectangle((0, 0), 1, 1, color=INK, alpha=0.5, label='burned-in text (masked)'))
    ax.legend(handles=handles, loc='lower right', fontsize=8, framealpha=0.9, labelcolor=INK)
    ax.set_title(f'Automatic choices: sides of the fault, regions, masks (reference frame {ref}; '
                 f'uncolored windows: periodic, flat or low SNR)', color=INK, fontsize=10, loc='left')
    ax.set_xlim(0, w)
    ax.set_ylim(h, 0)
    ax.tick_params(colors=INK2, labelsize=8)
    for sp in ax.spines.values():
        sp.set_visible(False)
    fig.savefig(args.out + '_map.png', dpi=110, facecolor='#fcfcfb')

    fig, axes = plt.subplots(2, 1, figsize=(11, 6.5), sharex=True, constrained_layout=True,
                             gridspec_kw=dict(height_ratios=(3, 2)))
    for r in results[:2]:
        c = colors[r['region']]
        mr = pulse_metrics(t, r['_series'], r['final_total_px'], args.smooth)
        axes[0].fill_between(t, r['_lo'], r['_hi'], color=c, alpha=0.18, lw=0)
        axes[0].plot(t, r['_series'], color=c, lw=1.8, label=f'region {r["region"]}: {r["windows"]} windows, '
                     f'final {r["final_total_px"]:.1f} px')
        axes[1].plot(t, mr['rate'], color=c, lw=1.6)
    axes[0].set_ylabel('Slip, pixels (band: 16-84 % of windows)', color=INK2, fontsize=9)
    axes[0].legend(frameon=False, fontsize=9, loc='upper left', labelcolor=INK)
    axes[0].set_title('Automatic slip: far-side regions relative to their paired camera-side windows', color=INK,
                      fontsize=11, loc='left')
    axes[1].set_ylabel(f'Slip rate, pixels/s ({args.smooth}-frame smoothing)', color=INK2, fontsize=9)
    axes[1].set_xlabel(f'Time after the reference frame {ref}, s', color=INK2, fontsize=9)
    for ax in axes:
        ax.axhline(0, color=INK2, lw=0.8)
        ax.grid(color=GRID, lw=0.6)
        ax.tick_params(colors=INK2, labelsize=8)
        for sp in ax.spines.values():
            sp.set_visible(False)
    t0 = gm['t10']
    tmin, tmax = max(t[0], t0 - 2), min(t[-1], t0 + 4)
    axes[1].set_xlim(tmin, tmax)
    shown = (t >= tmin) & (t <= tmax)
    rates = np.concatenate([pulse_metrics(t, r['_series'], r['final_total_px'], args.smooth)['rate'][shown]
                            for r in results[:2]]) if results else np.zeros(1)
    lo, hi = np.nanmin(rates), np.nanmax(rates)
    axes[1].set_ylim(min(lo, 0) - 0.1 * (hi - lo), hi + 0.15 * (hi - lo))
    fig.savefig(args.out + '_slip.png', dpi=110, facecolor='#fcfcfb')
    print(f'-> {args.out}_summary.json, _map.png, _slip.png, _region*.csv, _offsets.npz')


if __name__ == '__main__':
    main()
