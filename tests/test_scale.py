"""
Tests of slipcam.scale: the scale of an oblique camera looking at the ground, recovered from control
points and from scale bars

usage: python -m pytest tests
"""

import numpy as np
import pytest

from slipcam import scale as scales


def camera(height=8.0, tilt=20.0, heading=30.0, focal=1000.0, size=(720, 1280)):
    """A pinhole camera height m above the ground, looking down by tilt degrees toward the azimuth
    heading: the homography of the ground (east, north) onto the image (x, y)"""
    t, a = np.radians(tilt), np.radians(heading)
    forward = np.array([np.sin(a) * np.cos(t), np.cos(a) * np.cos(t), -np.sin(t)])
    right = np.array([np.cos(a), -np.sin(a), 0.0])
    down = np.cross(forward, right)
    R = np.stack([right, down, forward])                  # world to camera: x right, y down, z forward
    K = np.array([[focal, 0, size[1] / 2], [0, focal, size[0] / 2], [0, 0, 1]])
    c = np.array([0, 0, height])
    P = K @ np.column_stack([R, -R @ c])                   # 3 x 4
    return P[:, [0, 1, 3]]                                 # the plane z = 0


def to_ground(G2I, x, y):
    return scales.project(np.linalg.inv(G2I), x, y)


@pytest.fixture
def cam():
    return camera()


def test_control(cam):
    rng = np.random.default_rng(0)
    xy = np.column_stack([rng.uniform(100, 1180, 6), rng.uniform(450, 700, 6)])
    en = np.column_stack(to_ground(cam, *xy.T))
    s = scales.from_control(np.column_stack([xy, en]))
    # the ground displacements of image displacements, and the Jacobian, anywhere in the ground part of the image
    for x, y in ((640, 600), (200, 500), (1100, 690)):
        d = np.array([30.0, -20.0])
        expect = np.subtract(to_ground(cam, x + d[0], y + d[1]), to_ground(cam, x, y))
        assert np.allclose(s.ground(x, y, d), expect, rtol=1e-6)
        h = 1e-3
        fd = np.column_stack([np.subtract(to_ground(cam, x + h, y), to_ground(cam, x - h, y)),
                              np.subtract(to_ground(cam, x, y + h), to_ground(cam, x, y - h))]) / (2 * h)
        assert np.allclose(s.jacobian(x, y), fd, rtol=1e-5)
    # a scale not uniform: farther up the image, more meters per pixel, and more down than across
    assert s.along(640, 450, [0, 1]) > 2 * s.along(640, 690, [0, 1])
    assert s.along(640, 600, [0, 1]) > 1.5 * s.along(640, 600, [1, 0])


def test_control_errors(cam):
    with pytest.raises(ValueError):
        scales.from_control([[0, 0, 0, 0]] * 3)
    with pytest.raises(ValueError):
        scales.from_control([[0, 0, 0, 0], [1, 1, 1, 1], [2, 2, 2, 2], [3, 3, 3, 3]])     # on a line


def test_bars(cam):
    x, y = 640, 600
    J = np.linalg.inv(scales.homography_jacobian(cam, *to_ground(cam, x, y)))     # m per pixel, at (x, y)
    dirs = np.array([[40, 0], [0, 25], [30, 30], [-20, 35]], float)
    bars = [(x, y, x + v[0], y + v[1], np.linalg.norm(J @ v)) for v in dirs]
    test = np.array([[1, 0], [0, 1], [1, -2], [3, 1]], float)
    truth = np.linalg.norm(test @ J.T, axis=1) / np.linalg.norm(test, axis=1)
    # three or more bars: the lengths in every direction
    for n in (3, 4):
        assert np.allclose(scales.from_bars(bars[:n]).along(x, y, test), truth, rtol=1e-6)
    # two, along the rows and the columns: those two scales
    two = scales.from_bars(bars[:2])
    assert np.allclose(two.along(x, y, test[:2]), truth[:2], rtol=1e-6)
    # one: its own direction
    one = scales.from_bars(bars[:1])
    assert np.isclose(one.along(x, y, [1, 0]), truth[0]) and np.isclose(one.along(x, y, [0, 1]), truth[0])
    with pytest.raises(ValueError):
        scales.from_bars([bars[0], (x, y, x + 80, y + 1, 1.0)])            # two in the same direction


def test_slip(cam):
    """A known slip on the ground, seen at windows over the image, recovered with control points"""
    rng = np.random.default_rng(1)
    xy = np.column_stack([rng.uniform(100, 1180, 8), rng.uniform(450, 700, 8)])
    s = scales.from_control(np.column_stack([xy, np.column_stack(to_ground(cam, *xy.T))]))
    slip = np.array([2.0, 1.0])                                            # east, north, m
    x, y = np.meshgrid(np.linspace(500, 800, 5), np.linspace(550, 650, 4))
    x, y = x.ravel(), y.ravel()
    e, n = to_ground(cam, x, y)
    x1, y1 = scales.project(cam, e + slip[0], n + slip[1])
    ramp = np.clip((np.arange(60) - 20) / 10, 0, 1)                         # frames
    d = ramp[:, None, None] * np.stack([x1 - x, y1 - y], axis=-1)[None]     # frames x windows x 2
    fin = np.arange(60) >= 40
    g = scales.slip(s, x, y, d, fin)
    # a slip of ~100 px, through the homography
    assert np.allclose(g['final'], slip, rtol=1e-6)
    assert np.isclose(g['azimuth'], np.degrees(np.arctan2(*slip)))
    assert np.allclose(g['along'][fin], np.hypot(*slip), rtol=1e-6)
    assert np.all(np.abs(g['along'][:20]) < 1e-9)
