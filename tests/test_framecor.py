"""
Tests of framecor: known shifts of a synthetic texture, recovered by both backends

usage: python -m pytest tests
"""

import numpy as np
import pytest
from scipy import ndimage

from slipcam import framecor

BACKENDS = ['cpu'] + (['cuda'] if framecor.has_cuda() else [])


def texture(h=240, w=320, seed=0):
    """A random texture, smoothed (a correlation length of a few pixels), in 20-220"""
    img = ndimage.gaussian_filter(np.random.default_rng(seed).normal(size=(h, w)), 2.0)
    return (img - img.min()) / (img.max() - img.min()) * 200 + 20


def frames(shifts, h=240, w=320):
    """The reference frame and the frames of the texture moved by the shifts (down, across)"""
    base = texture(h, w)
    to8 = lambda img: np.clip(np.rint(img), 0, 255).astype(np.uint8)     # noqa: E731
    return to8(base), np.stack([to8(ndimage.shift(base, s, order=3, mode='reflect')) for s in shifts])


def correlator(backend, window=32, search=24, skip=24, h=240, w=320, margin=0):
    down, across = framecor.window_grid(h, w, window, search, skip, margin=margin)
    return framecor.Correlator(h, w, down, across, window=window, search=search, backend=backend)


@pytest.mark.parametrize('backend', BACKENDS)
def test_shifts(backend):
    shifts = [(0, 0), (0.3, -0.7), (5.25, -12.5), (-20.4, 15.1)]
    ref, sec = frames(shifts)
    cor = correlator(backend)
    assert cor.backend == backend
    cor.set_reference(ref)
    r = cor.correlate(sec)
    assert r['offsets'].shape == (len(shifts),) + cor.shape + (2,)
    for k, s in enumerate(shifts):
        err = r['offsets'][k] - np.array(s, np.float32)
        assert np.all(np.isfinite(err))
        assert np.median(np.abs(err)) < 0.05, (s, np.median(np.abs(err), axis=(0, 1)))
        assert np.abs(err).max() < 0.2, (s, np.abs(err).max())
        assert np.median(r['snr'][k]) > 10
        assert np.median(r['peak'][k]) > 0.9


@pytest.mark.skipif('cuda' not in BACKENDS, reason='no CUDA backend')
def test_backends_agree():
    ref, sec = frames([(0.3, -0.7), (5.25, -12.5), (-20.4, 15.1)])
    out = {}
    for backend in ('cpu', 'cuda'):
        cor = correlator(backend)
        cor.set_reference(ref)
        out[backend] = cor.correlate(sec)
    for key, tol in (('offsets', 1e-3), ('peak', 1e-4)):
        assert np.abs(out['cpu'][key] - out['cuda'][key]).max() < tol, key
    assert np.abs(out['cpu']['snr'] / out['cuda']['snr'] - 1).max() < 1e-2


@pytest.mark.parametrize('backend', BACKENDS)
def test_gross(backend):
    """A motion larger than the search range, found with gross offsets"""
    shift = (30.4, -27.6)
    ref, sec = frames([shift])
    cor = correlator(backend, search=6, skip=16, margin=32)
    cor.set_reference(ref)
    found = cor.correlate(sec, gross=np.array([[30, -28]]))['offsets'][0]
    assert np.median(np.abs(found - np.array(shift, np.float32))) < 0.05
    # per window: the same
    g = np.broadcast_to(np.array([30, -28], np.int32), (1,) + cor.shape + (2,))
    assert np.array_equal(cor.correlate(sec, gross=g)['offsets'][0], found, equal_nan=True)
    # without: wrong
    assert np.median(np.abs(cor.correlate(sec)['offsets'][0] - np.array(shift, np.float32))) > 1


@pytest.mark.parametrize('backend', BACKENDS)
def test_invalid(backend):
    """Search windows outside the frame, and flat windows, are not valid"""
    ref, sec = frames([(0, 0)])
    cor = correlator(backend)
    cor.set_reference(ref)
    r = cor.correlate(sec, gross=np.array([[0, 1000]]))
    assert np.all(np.isnan(r['offsets'])) and np.all(r['snr'] == 0)

    flat = ref.copy()
    flat[:120] = 100
    cor.set_reference(flat)
    r = cor.correlate(sec)
    assert np.all(np.isnan(r['offsets'][0, 0]))          # the first row of windows: flat
    assert np.all(np.isfinite(r['offsets'][0, -1]))


def test_arguments():
    with pytest.raises(ValueError):
        framecor.Correlator(240, 320, [0], [300], window=32, search=8, backend='cpu')    # outside
    with pytest.raises(ValueError):
        framecor.Correlator(240, 320, [0], [0], window=32, search=8, backend='gpu')
    cor = correlator('cpu')
    with pytest.raises(RuntimeError):
        cor.correlate(np.zeros((1, 240, 320), np.uint8))                                 # no reference
    cor.set_reference(np.zeros((240, 320), np.uint8))
    with pytest.raises(ValueError):
        cor.correlate(np.zeros((1, 240, 321), np.uint8))
