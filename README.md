# slipcam

Surface fault slip from video: dense image correlation of the frames on the GPU (pycuampcor), the
camera motion removed by pairing targets across the fault, and the slip, its duration and its peak
rate, in pixels.

## Requirements

* Python with numpy, scipy, matplotlib
* [pycuampcor](https://github.com/earthdef/cuAmpcor) (2.3 or later, built with CUDA; `--cpu` in
  `track_offsets.py` for the CPU implementation)
* ffmpeg, to decode the videos (any build; e.g., the one bundled with `imageio-ffmpeg`)

The scripts import each other: run them from this directory, or add it to `PYTHONPATH`.

## Usage

```bash
# frames (uint8 .npy) and the timing of the video, from the burned-in clock if any
python decode_frames.py video.mp4 --ffmpeg $(which ffmpeg) --out run/frames.npy --clock 1185 28 1235 64
python decode_frames.py news.mp4 --out run/clip.npy --start 25.71 --duration 10.67      # a clip of a video

# the automatic pipeline: frames to slip, outputs run/auto_*
python auto_slip.py run/frames.npy --fps 30 --out run/auto
python auto_slip.py run/frames.npy --fps 30 --out run/auto --explain 980 130 1220 200   # why a box was (not) used
```

Outputs of `auto_slip.py` (prefix `--out`): `_summary.json` (timing, sides, regions, slip, rise, peak
rate), `_region*.csv` (the slip series), `_map.png` (the sides and regions), `_slip.png` (slip and rate),
and the caches `_offsets.npz`, `_pairs.npz` (and `_stabilized.npy`, `_stabilization.npz`).

The manual steps, for exploring a video or checking the automatic choices:

```bash
python track_offsets.py run/frames.npy --ref 285 --first 140 --out run/offsets.npz \
    --mask 870 25 1240 68 --mask 40 650 630 700                     # boxes of burned-in text
python offset_map.py run/offsets.npz run/frames.npy --frames 770 806 --vmax 30 --out run/final_map.png
python region_series.py run/offsets.npz --region fence 1000 210 1220 350 --region field 980 130 1220 200 \
    --tmin 3 --tmax 9 --out run/series
python slip_series.py run/offsets.npz run/frames.npy --model shift --near 1000 210 1220 350 \
    --far 980 130 1220 200 --final 600 700 --smooth 5 --final-slip 2.5 --out run/slip
```

## Scripts

| Script | What it does |
|---|---|
| `decode_frames.py` | video to grayscale frames; frozen and duplicated frames; the ticks of the burned-in clock |
| `track_offsets.py` | dense offsets of the frames relative to a reference frame, with pycuampcor |
| `offset_map.py` | maps of the offsets over the frame |
| `region_series.py` | time series of the median offsets of boxes |
| `slip_series.py` | the slip of a far-side box relative to a camera-side box; rise, duration, peak rate |
| `stabilize.py` | coarse tracking and warping of the frames onto a reference (similarity transform) |
| `auto_slip.py` | the automatic pipeline (below) |

## The automatic pipeline

1. **timing**: frozen frames; the onset of the shaking from the differences of consecutive frames,
   relative to the first calm frames; the reference, the last calm frame;
2. **text**: burned-in text, strong edges that do not change while the scene shakes, masked;
3. **stabilization** (if there are no calm frames, or `--stabilize on`): the frames warped onto the
   reference;
4. **tracking**: pycuampcor on the GPU;
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
background from an analyst. The offsets are in pixels: the scale to meters (the distance of the target,
objects of known size) is not measured.
