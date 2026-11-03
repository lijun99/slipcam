# slipcam

Surface fault slip from video: dense image correlation of the frames on the GPU (framecor), the camera
motion removed by pairing targets across the fault, and the slip, its duration and its peak rate, in
pixels or, with a scale, in meters.

## Install

In an environment with numpy, scipy, matplotlib, pybind11, scikit-build-core, FFTW (single precision)
and, for the GPU, the CUDA toolkit (e.g., the isce3 conda environment):

```bash
CUDACXX=/usr/local/cuda/bin/nvcc pip install . --no-deps --no-build-isolation \
    -Ccmake.define.CMAKE_CUDA_ARCHITECTURES=native
pip install . --no-deps --no-build-isolation -Ccmake.define.FRAMECOR_CUDA=OFF     # CPU only
python -m pytest tests
```

Also needed: ffmpeg, to decode the videos (any build; e.g., the one bundled with `imageio-ffmpeg`).

## Usage

```bash
# frames (uint8 .npy) and the timing of the video, from the burned-in clock if any
slipcam-decode video.mp4 --ffmpeg $(which ffmpeg) --out run/frames.npy --clock 1185 28 1235 64
slipcam-decode news.mp4 --out run/clip.npy --start 25.71 --duration 10.67      # a clip of a video

# the automatic pipeline: frames to slip, outputs run/auto_*
slipcam-auto run/frames.npy --fps 30 --out run/auto
slipcam-auto run/frames.npy --fps 30 --out run/auto --explain 980 130 1220 200   # why a box was (not) used
```

Outputs of `slipcam-auto` (prefix `--out`): `_summary.json` (timing, sides, regions, slip, rise, peak
rate), `_region*.csv` (the slip series), `_map.png` (the sides and regions), `_slip.png` (slip and rate),
and the caches `_offsets.npz`, `_pairs.npz` (and `_stabilized.npy`, `_stabilization.npz`).

The manual steps, for exploring a video or checking the automatic choices:

```bash
slipcam-track run/frames.npy --ref 285 --first 140 --out run/offsets.npz \
    --mask 870 25 1240 68 --mask 40 650 630 700                     # boxes of burned-in text
slipcam-map run/offsets.npz run/frames.npy --frames 770 806 --vmax 30 --out run/final_map.png
slipcam-regions run/offsets.npz --region fence 1000 210 1220 350 --region field 980 130 1220 200 \
    --tmin 3 --tmax 9 --out run/series
slipcam-slip run/offsets.npz run/frames.npy --model shift --near 1000 210 1220 350 \
    --far 980 130 1220 200 --final 600 700 --smooth 5 --final-slip 2.5 --out run/slip
```

## Modules

| Command | Module | What it does |
|---|---|---|
| `slipcam-decode` | `slipcam.decode_frames` | video to grayscale frames; frozen and duplicated frames; the ticks of the burned-in clock |
| `slipcam-track` | `slipcam.track_offsets` | dense offsets of the frames relative to a reference frame, with framecor |
| `slipcam-map` | `slipcam.offset_map` | maps of the offsets over the frame |
| `slipcam-regions` | `slipcam.region_series` | time series of the median offsets of boxes |
| `slipcam-slip` | `slipcam.slip_series` | the slip of a far-side box relative to a camera-side box; rise, duration, peak rate |
| | `slipcam.stabilize` | coarse tracking and warping of the frames onto a reference (similarity transform) |
| | `slipcam.scale` | the scale of the image, m per pixel (below) |
| `slipcam-auto` | `slipcam.auto_slip` | the automatic pipeline (below) |
| | `slipcam.framecor` | the correlator (below) |

Each command is also `python -m slipcam.<module>`.

Layout: `python/slipcam` (the package), `cxx` (the C++ and CUDA of framecor), `tests`.

## framecor

framecor replaces [pycuampcor](https://github.com/earthdef/cuAmpcor), made for SAR images (complex, two
passes with oversampled chips, files in and out, one run per pair). It is a normalized
cross-correlation for 8-bit frames, in C++ with a CUDA backend (cuFFT) and a CPU backend (FFTW,
OpenMP), and a pybind11 module. A grid of square windows of the reference frame is searched in each
frame within +/- `search` pixels (plus optional integer gross offsets, per frame or per window). For
each window:

* the correlation with the zero-mean reference window, by real FFTs in batches of windows (of one or
  more frames, in memory), the spectra of the reference windows computed once;
* normalized by the variance of the frame under each shift, from summed-area tables (exact integer
  sums of the 8-bit pixels);
* the peak refined to sub-pixel by a quadratic fitted to the 3 x 3 surface around it (a single pass);
* the SNR and the covariance of the offsets as in pycuampcor.

Offsets are (down, across): the motion of the frame relative to the reference frame, as in pycuampcor.

```python
from slipcam import framecor

down, across = framecor.window_grid(720, 1280, window=64, search=48, skip=16)
cor = framecor.Correlator(720, 1280, down, across, window=64, search=48, backend='auto')
cor.set_reference(frames[ref])                       # uint8, height x width
r = cor.correlate(frames[first:last])                # uint8, frames x height x width
r['offsets'], r['peak'], r['snr'], r['cov']          # frames x down x across (x 2, x 3)
r = cor.correlate(frames[first:last], gross=g)       # g: frames x 2, or frames x down x across x 2
```

Compared with pycuampcor 2.3 on the Myanmar video (2450 windows of 64 x 64 pixels, +/- 48 pixels), on
an RTX 4060 laptop GPU and a 14900HX (16 threads):

| | ms per frame |
|---|---|
| pycuampcor, GPU | 160 |
| framecor, GPU | 22 |
| framecor, CPU | 143 |

and the same results: the same SNR distribution; sub-pixel differences of 0.02-0.03 px (MAD); the manual
slip 35.47 px vs 35.45 px (rise 0.90 s both); the automatic primary region 39.5 px both (peak rate 46.7
vs 47.4 px/s). The integer peaks differ for 10 % of the windows with SNR >= 4 (2 % before the shaking, 13
% after it, mostly by 2-4 px at lower correlation, 0.73 vs 0.86): ambiguous surfaces, where pycuampcor's
second pass moves to another local peak within its zoom window.

## The scale

The scale of a camera is not uniform: meters per pixel grow with the distance, and for an oblique view
the ground is foreshortened along the columns of the image (a pixel down covers more ground than a
pixel across). The slip is in pixels unless a scale is given, in the reference frame:

| Option | Scale | When |
|---|---|---|
| `--final-slip M` (`slipcam-slip`) | uniform, calibrated to a measured final slip | a field measurement of the same target |
| `--scale M` | uniform, m per pixel | a rough value |
| `--scale-bar X0 Y0 X1 Y1 M` | local: 1 bar, isotropic; 2, along the rows and the columns; 3 or more, any direction | objects of known size next to the target (fence panels, a car, lane markings, a door) |
| `--control X Y E N` (4 or more) | a homography onto the ground plane: a scale at each window, and the azimuth of the slip | points also in an orthophoto, a satellite image or a survey |

With control points each window is converted at its own scale (through the homography, not
linearized); with scale bars, at theirs. The slip is reported along its final direction on the ground
(in the summary of `slipcam-auto`: `final_total_m`, `peak_rate_m_s`, `scale_cm_px`, `azimuth_deg`).

The slip is taken to be in the plane of the scale: with one camera, vertical motion cannot be told from
horizontal motion along the line of sight. A target above the ground (a wall, a fence) is nearer than
the ground behind it in the image: give bars or control points at its height. The lens distortion of
wide-angle cameras is not modeled: use control points around the target. Scale bars give one scale for
the image, so they are good only near the target; for several targets, use control points.

```bash
slipcam-slip run/offsets.npz run/frames.npy ... --scale-bar 1000 165 1100 165 7.0     # a 7 m fence, 100 px
slipcam-auto run/frames.npy --out run/auto --control 212 640 0 0 --control 1180 655 18.2 3.1 \
    --control 1015 240 31.0 52.4 --control 300 260 -12.5 47.9           # e.g., E, N, m, from an orthophoto
```

## The automatic pipeline

1. **timing**: frozen frames; the onset of the shaking from the differences of consecutive frames,
   relative to the first calm frames; the reference, the last calm frame;
2. **text**: burned-in text, strong edges that do not change while the scene shakes, masked;
3. **stabilization** (if there are no calm frames, or `--stabilize on`): the frames warped onto the
   reference;
4. **tracking**: framecor on the GPU;
5. **quality**: low SNR dropped; periodic or flat textures (a second peak of their correlation with their
   own surroundings) flagged;
6. **sides**: the residuals of the offsets just after the slip to a robust fit of the camera motion; the
   far side, consistent and stable residuals above 4 robust deviations;
7. **camera motion**: each far-side window paired with the camera-side windows near it whose motion
   before the slip matches its own (not those next to the far side, whose templates overlap it);
8. **slip**: regions of similar far-side offsets, scored by how step-like their slip is; the primary
   region, the largest slip of those scoring 5 or more.

## Status

Tested on the three videos of Hirano (2026), without retuning:

| Video | Result | Published |
|---|---|---|
| 2025 Mw 7.7 Myanmar, fixed CCTV | **works**: 35.5 px (manual pair), 39.5 px (automatic primary region, another target); rise 0.90 s; peak 39-47 px/s | 35.5 px, 39.4 px/s (Hirano 2026); 2.5 m, 1.3 s, 3.2 m/s (Kearse & Kaneko 2025) |
| 2018 Hualien, CCTV in a news report | **partly**: the clip starts during the shaking; regions of 20-26 px, peak ~40-42 px/s; the sides are suspect (the camera side is under the captions) | 42.1 px, 41.8 px/s |
| 2026 Kumamoto, a phone filming a CCTV screen | **fails**: the stabilization locks onto a moving truck, and the phone needs a perspective transform | 26.2 px |

The pipeline is automatic for fixed cameras with calm frames before the shaking. Broadcast and phone
recordings need a robust stabilization (on the background only, with a homography), or a box of stable
background from an analyst. The scale to meters is not measured from the video itself: it needs scale
bars or control points (above).

## License

MIT, copyright California Institute of Technology; see [LICENSE](LICENSE).
