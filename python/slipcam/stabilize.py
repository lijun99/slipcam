#!/usr/bin/env python3
"""
Stabilization of video frames on a reference frame, for cameras (or phones filming a screen) that move
more than the search range of the dense tracking

A coarse grid of large windows of the reference frame (away from the burned-in text) is tracked in
each frame by normalized cross-correlation, around the positions predicted by the transform of the
previous frame, so that a large total motion needs only a small search; a robust fit of a similarity
transform (shift, rotation, scale) to the positions gives the transform of each frame relative to the
reference, without accumulating errors. The frames are then warped onto the reference, and their
burned-in text (fixed in the recorded frames, so moving in the warped ones) is blanked.

needs: numpy, scipy
"""

import numpy as np
from scipy import ndimage, signal

from .slip_series import fit_camera


def ncc_peak(template, search):
    """The position (dy, dx) of the peak of the normalized cross-correlation of template in search,
    relative to the center of search, with a parabolic sub-pixel refinement, and the peak value"""
    t0 = template - template.mean()
    tn = np.sqrt((t0 ** 2).sum())
    if tn < 1e-6:
        return np.nan, np.nan, 0.0
    ones = np.ones(template.shape)
    num = signal.fftconvolve(search, t0[::-1, ::-1], mode='valid')
    ssum = signal.fftconvolve(search, ones, mode='valid')
    ssq = signal.fftconvolve(search ** 2, ones, mode='valid')
    var = np.maximum(ssq - ssum ** 2 / template.size, 1e-6)
    ncc = num / (np.sqrt(var) * tn)
    i, j = np.unravel_index(np.argmax(ncc), ncc.shape)
    peak = ncc[i, j]

    def refine(c, m, p):
        den = m - 2 * c + p
        return 0.0 if abs(den) < 1e-12 else 0.5 * (m - p) / den
    di = refine(ncc[i, j], ncc[i - 1, j], ncc[i + 1, j]) if 0 < i < ncc.shape[0] - 1 else 0.0
    dj = refine(ncc[i, j], ncc[i, j - 1], ncc[i, j + 1]) if 0 < j < ncc.shape[1] - 1 else 0.0
    c0, c1 = (ncc.shape[0] - 1) / 2, (ncc.shape[1] - 1) / 2
    return i + di - c0, j + dj - c1, peak


def apply(params, x, y):
    """The positions of the points (x, y), relative to the image center, under the similarity transform
    params = (p, q, a, s): x' = x + p - a y + s x, y' = y + q + a x + s y"""
    p, q, a, s = params
    return x + p - a * y + s * x, y + q + a * x + s * y


def coarse_track(frames, ref, text, window=96, step=96, search=32, min_peak=0.5, start=0):
    """The similarity transform of each frame relative to the frame ref (frames x 4: p, q, a, s, as in
    slip_series.fit_camera, about the image center), the number of windows used, and their grid

    The windows of the grid that overlap the text mask (or are flat) are not used; each frame is
    tracked from the transform of the previous one (forward from ref, then backward)."""
    n, h, w = frames.shape
    half = window // 2
    yc = np.arange(search + half, h - search - half, step)
    xc = np.arange(search + half, w - search - half, step)
    Yg, Xg = [v.ravel() for v in np.meshgrid(yc, xc, indexing='ij')]
    refimg = frames[ref].astype(np.float64)
    keep = np.array([not text[y - half:y + half, x - half:x + half].any() and
                     refimg[y - half:y + half, x - half:x + half].std() > 2 for y, x in zip(Yg, Xg)])
    Yg, Xg = Yg[keep], Xg[keep]
    templates = [refimg[y - half:y + half, x - half:x + half] for y, x in zip(Yg, Xg)]
    X0, Y0 = Xg - w / 2, Yg - h / 2                                     # relative to the image center

    params = np.full((n, 4), np.nan)
    used = np.zeros(n, int)
    params[ref] = 0
    used[ref] = len(Xg)
    for order in (range(ref + 1, n), range(ref - 1, start - 1, -1)):
        prev = np.zeros(4)
        for k in order:
            img = frames[k].astype(np.float64)
            px, py = apply(prev, X0, Y0)
            dx = np.full(len(Xg), np.nan)
            dy = np.full(len(Xg), np.nan)
            for m, (tpl, cx, cy) in enumerate(zip(templates, px + w / 2, py + h / 2)):
                y0, x0 = int(round(cy)) - half - search, int(round(cx)) - half - search
                if y0 < 0 or x0 < 0 or y0 + window + 2 * search > h or x0 + window + 2 * search > w:
                    continue
                oy, ox, peak = ncc_peak(tpl, img[y0:y0 + window + 2 * search, x0:x0 + window + 2 * search])
                if peak >= min_peak:
                    dx[m] = (x0 + search + half + ox) - w / 2 - X0[m]
                    dy[m] = (y0 + search + half + oy) - h / 2 - Y0[m]
            fit, nuse = fit_camera(dx, dy, X0, Y0)
            if np.all(np.isfinite(fit)):
                params[k], used[k], prev = fit, nuse, fit
            else:
                params[k], used[k] = prev, 0
    return params, used, len(Xg)


def warp(frame, params, fill=None):
    """The frame resampled onto the reference: the value at (x, y) of the reference is that of the frame
    at the transformed position (bilinear); outside the frame, fill (default: its mean)"""
    h, w = frame.shape
    yy, xx = np.mgrid[0:h, 0:w].astype(np.float64)
    sx, sy = apply(params, xx - w / 2, yy - h / 2)
    fill = frame.mean() if fill is None else fill
    return ndimage.map_coordinates(frame.astype(np.float32), [sy + h / 2, sx + w / 2], order=1, cval=fill)


def stabilize(frames, params, text):
    """The frames warped onto the reference (uint8), with their text blanked (the local mean), and the
    union of the warped text masks"""
    n, h, w = frames.shape
    out = np.empty_like(frames)
    union = np.zeros((h, w), bool)
    keep = (~text).astype(np.float32)
    den = ndimage.uniform_filter(keep, 31)
    for k in range(n):
        f = frames[k].astype(np.float32)
        if text.any():
            local = ndimage.uniform_filter(f * keep, 31) / np.maximum(den, 1e-3)
            f = np.where(text, np.where(den > 1e-3, local, f[~text].mean()), f)
        out[k] = np.clip(warp(f, params[k]), 0, 255).astype(np.uint8)
        union |= warp(text.astype(np.float32), params[k], fill=0) > 0.5
    return out, union
