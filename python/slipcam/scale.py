"""
The scale of the image, meters on the ground per pixel: not uniform, as it grows with the distance from
the camera and, for an oblique view, depends on the direction (the ground recedes along the columns of
the image, foreshortened)

A scale is a Jacobian J (x, y), m per pixel, of the ground coordinates with respect to those of the
image: a displacement d = (across, down), pixels, at (x, y) of the reference frame is J d on the ground.
From:

* a uniform scale (--scale, m per pixel), the same everywhere and in every direction;
* scale bars (--scale-bar X0 Y0 X1 Y1 LENGTH), known lengths near the target (e.g., a fence panel, a
  car, a lane marking, a building): one bar, a scale the same in every direction; two, different scales
  along the rows and the columns of the image; three or more in different directions, the lengths in
  every direction (the local metric); the same over the image, so the bars must be near the target;
* ground control points (--control X Y E N, four or more), points of the ground of known coordinates
  (e.g., from a survey, or features also in an orthophoto or a satellite image): a homography of the
  image onto the ground plane, the displacements mapped through it (not linearized: a slip of tens of
  pixels spans a range of scales); with east and north, the azimuth of the slip.

The motion is taken to be in the plane of the scale (horizontal, for the ground): with one camera,
vertical motion cannot be told apart from horizontal motion along the line of sight. A target above the
ground (a wall, a fence) is nearer the camera than the ground behind it in the image: use bars or
control points at its height. The lens distortion (the barrel of wide-angle CCTV) is not modeled: use
control points around the target.
"""

import warnings

import numpy as np


class Scale:
    """A scale: the Jacobian J (x, y), m per pixel, mapping (across, down) pixels to the ground, uniform
    (J) or of a homography (H) of the image onto the ground; info, its description"""

    def __init__(self, J=None, H=None, info='', azimuth=False):
        self.J, self.H, self.info, self.azimuth = J, H, info, azimuth

    def jacobian(self, x, y):
        x, y = np.broadcast_arrays(np.asarray(x, float), np.asarray(y, float))
        if self.H is None:
            return np.broadcast_to(self.J, x.shape + (2, 2))
        return homography_jacobian(self.H, x, y)

    def ground(self, x, y, d):
        """The ground displacements (..., 2), m, of the image displacements d (..., 2: across, down),
        pixels, from (x, y) (broadcast against the leading dimensions of d); through the homography, not
        linearized (a slip of tens of pixels spans a range of scales)"""
        d = np.asarray(d, float)
        if self.H is None:
            return np.einsum('...ij,...j->...i', self.jacobian(x, y), d)
        e0, n0 = project(self.H, np.asarray(x, float), np.asarray(y, float))
        e1, n1 = project(self.H, x + d[..., 0], y + d[..., 1])
        return np.stack([e1 - e0, n1 - n0], axis=-1)

    def along(self, x, y, d):
        """The scale along the image directions d (..., 2: across, down), m per pixel"""
        d = np.asarray(d, float)
        return np.linalg.norm(self.ground(x, y, d / np.linalg.norm(d, axis=-1, keepdims=True)), axis=-1)


def uniform(scale):
    """A uniform scale, m per pixel"""
    return Scale(J=scale * np.eye(2), info=f'uniform scale {scale * 100:.2f} cm/px')


def from_bars(bars):
    """The local metric G (J = G^1/2) of scale bars (x0, y0, x1, y1, length): one bar, isotropic; two,
    without shear (the rows and the columns of the image); three or more, least squares"""
    b = np.asarray(bars, float).reshape(-1, 5)
    v, length = b[:, 2:4] - b[:, :2], b[:, 4]
    if np.any(np.hypot(*v.T) == 0) or np.any(length <= 0):
        raise ValueError('scale bars: a bar of no length')
    A = np.column_stack([v[:, 0] ** 2, 2 * v[:, 0] * v[:, 1], v[:, 1] ** 2])
    if len(b) == 1:
        g = np.array([1, 0, 1]) * (length[0] / np.hypot(*v[0])) ** 2
    elif len(b) == 2:
        A2 = A[:, [0, 2]]
        if np.linalg.cond(A2) > 1e3:
            raise ValueError('scale bars: two bars in about the same direction; give bars along the rows and the columns')
        gxx, gyy = np.linalg.solve(A2, length ** 2)
        g = np.array([gxx, 0, gyy])
    else:
        if np.linalg.cond(A) > 1e3:
            raise ValueError('scale bars: the bars do not span three directions')
        g = np.linalg.lstsq(A, length ** 2, rcond=None)[0]
    G = np.array([[g[0], g[1]], [g[1], g[2]]])
    w, V = np.linalg.eigh(G)
    if w.min() <= 0:
        raise ValueError('scale bars: inconsistent lengths (no metric fits them)')
    J = V @ np.diag(np.sqrt(w)) @ V.T
    fit = np.sqrt(np.einsum('ni,ij,nj->n', v, G, v)) / length - 1
    info = (f'{len(b)} scale bar{"s" * (len(b) > 1)}: {np.sqrt(G[0, 0]) * 100:.2f} cm/px across, '
            f'{np.sqrt(G[1, 1]) * 100:.2f} cm/px down')
    if len(b) > 3:
        info += f'; bar lengths fitted within {np.abs(fit).max() * 100:.1f} %'
    return Scale(J=J, info=info)


def homography(src, dst):
    """The homography H (3 x 3) of the points src (n x 2) onto dst (n x 2), n >= 4, by the normalized DLT"""
    src, dst = np.asarray(src, float), np.asarray(dst, float)

    def normalization(p):
        c = p.mean(axis=0)
        s = np.sqrt(2) / np.mean(np.hypot(*(p - c).T))
        return np.array([[s, 0, -s * c[0]], [0, s, -s * c[1]], [0, 0, 1]])

    Ts, Td = normalization(src), normalization(dst)
    ps = np.column_stack([src, np.ones(len(src))]) @ Ts.T
    pd = np.column_stack([dst, np.ones(len(dst))]) @ Td.T
    A = []
    for (x, y, _), (u, v, _) in zip(ps, pd):
        A.append([0, 0, 0, -x, -y, -1, v * x, v * y, v])
        A.append([x, y, 1, 0, 0, 0, -u * x, -u * y, -u])
    _, sv, Vt = np.linalg.svd(np.array(A))
    if sv[-2] < 1e-9 * sv[0]:
        raise ValueError('control points: degenerate (three of them on a line?)')
    H = np.linalg.inv(Td) @ Vt[-1].reshape(3, 3) @ Ts
    return H / H[2, 2]


def project(H, x, y):
    """The points (x, y) mapped by H"""
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    return (H[0, 0] * x + H[0, 1] * y + H[0, 2]) / w, (H[1, 0] * x + H[1, 1] * y + H[1, 2]) / w


def homography_jacobian(H, x, y):
    """The Jacobian (..., 2, 2) of the map of H at (x, y)"""
    w = H[2, 0] * x + H[2, 1] * y + H[2, 2]
    u, v = project(H, x, y)
    return np.stack([np.stack([(H[0, 0] - u * H[2, 0]) / w, (H[0, 1] - u * H[2, 1]) / w], axis=-1),
                     np.stack([(H[1, 0] - v * H[2, 0]) / w, (H[1, 1] - v * H[2, 1]) / w], axis=-1)], axis=-2)


def from_control(points):
    """The homography of ground control points (x, y, east, north), pixels and m"""
    p = np.asarray(points, float).reshape(-1, 4)
    if len(p) < 4:
        raise ValueError(f'control points: {len(p)} given, 4 or more needed')
    H = homography(p[:, :2], p[:, 2:])
    w = H[2, 0] * p[:, 0] + H[2, 1] * p[:, 1] + H[2, 2]
    if np.ptp(np.sign(w)):
        raise ValueError('control points: on both sides of the horizon of their plane')
    e, n = project(H, p[:, 0], p[:, 1])
    rms = np.sqrt(np.mean((e - p[:, 2]) ** 2 + (n - p[:, 3]) ** 2))
    info = f'{len(p)} control points, a homography onto the ground'
    if len(p) > 4:
        info += f' (misfit {rms:.2f} m rms)'
    return Scale(H=H, info=info, azimuth=True)


def add_arguments(parser):
    """The options of the scale bars and the control points"""
    parser.add_argument('--scale-bar', type=float, nargs=5, action='append', default=[],
                        metavar=('X0', 'Y0', 'X1', 'Y1', 'M'),
                        help='a known length near the target: its ends in the reference frame, pixels, and\n'
                             'its length, m (repeatable: 1, isotropic; 2, along the rows and the columns;\n'
                             '3 or more, any directions)')
    parser.add_argument('--control', type=float, nargs=4, action='append', default=[], metavar=('X', 'Y', 'E', 'N'),
                        help='a ground control point: in the reference frame, pixels, and on the ground, m\n'
                             '(east, north; 4 or more; repeatable)')


def from_args(args):
    """The scale of the options of add_arguments, or None"""
    if args.scale_bar and args.control:
        raise SystemExit('give --scale-bar or --control, not both')
    if args.control:
        return from_control(args.control)
    if args.scale_bar:
        return from_bars(args.scale_bar)
    return None


def slip(scale, x, y, d, fin):
    """The slip on the ground of windows at (x, y) whose image offsets are d (frames x windows x 2:
    across, down), pixels: the final slip vector, m (the median over the windows of their medians over
    the frames fin), and, per frame, the median over the windows of the slip along it and normal to it"""
    g = scale.ground(x, y, d)
    with np.errstate(all='ignore'), warnings.catch_warnings():
        warnings.filterwarnings('ignore', 'All-NaN slice')    # frames without valid windows
        final = np.nanmedian(np.nanmedian(g[fin], axis=0), axis=0)
        u = final / np.hypot(*final)
        along = np.nanmedian(g @ u, axis=1)
        normal = np.nanmedian(g @ np.array([-u[1], u[0]]), axis=1)
    res = dict(final=final, direction=u, along=along, normal=normal, ground=g)
    if scale.azimuth:
        res['azimuth'] = float(np.degrees(np.arctan2(final[0], final[1])) % 360)
    return res
