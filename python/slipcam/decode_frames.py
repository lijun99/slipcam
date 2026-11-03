#!/usr/bin/env python3
"""
Decode a video into grayscale frames, and check its timing

The frames are decoded with ffmpeg into a uint8 array (frames x height x width), saved as .npy. The
timing is checked from the data, as re-uploaded videos may be slowed down or have duplicated frames:
the mean absolute difference of consecutive frames (0 for a duplicate), and the frames where the
burned-in clock of the camera changes (the box of its seconds digits), which mark the real seconds.

usage:
  slipcam-decode video.mp4 --ffmpeg /path/to/ffmpeg --out run/frames.npy --clock 1150 25 1235 65
  slipcam-decode news.mp4 --ffmpeg /path/to/ffmpeg --out run/clip.npy --start 25.5 --duration 10   # a clip
needs: numpy
"""

import argparse
import re
import subprocess

import numpy as np


def decode(video, ffmpeg, start=None, duration=None):
    """Read the size of the video, and decode its frames (from start, for duration, s) into a uint8 array"""
    info = subprocess.run([ffmpeg, '-hide_banner', '-i', video], capture_output=True, text=True).stderr
    stream = next(line for line in info.splitlines() if 'Video:' in line)
    width, height = (int(v) for v in re.search(r'\b(\d{2,5})x(\d{2,5})\b', stream).groups())
    trim = (['-ss', str(start)] if start is not None else []) + (['-t', str(duration)] if duration is not None else [])
    raw = subprocess.run([ffmpeg, '-hide_banner', '-loglevel', 'error', *trim, '-i', video,
                          '-f', 'rawvideo', '-pix_fmt', 'gray', '-'], capture_output=True, check=True).stdout
    return np.frombuffer(raw, np.uint8).reshape(-1, height, width)


def main():
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawTextHelpFormatter)
    parser.add_argument('video', help='the video file')
    parser.add_argument('--ffmpeg', default='ffmpeg', help='the ffmpeg executable (default: %(default)s)')
    parser.add_argument('--out', required=True, help='the output .npy of the frames')
    parser.add_argument('--start', type=float, help='the start of the clip in the video, s (default: the beginning)')
    parser.add_argument('--duration', type=float, help='the duration of the clip, s (default: to the end)')
    parser.add_argument('--fps', type=float, default=30, help='the frame rate of the video file (default: %(default)s)')
    parser.add_argument('--clock', type=int, nargs=4, metavar=('X0', 'Y0', 'X1', 'Y1'),
                        help='the box of the seconds digits of the burned-in clock, in pixels')
    args = parser.parse_args()

    frames = decode(args.video, args.ffmpeg, args.start, args.duration)
    np.save(args.out, frames)
    n, h, w = frames.shape
    print(f'{n} frames of {h} x {w} pixels, {n / args.fps:.2f} s at {args.fps:g} fps -> {args.out}')

    f = frames.astype(np.float32)
    diff = np.abs(np.diff(f, axis=0)).mean(axis=(1, 2))
    dup = diff < 0.5
    print(f'consecutive differences: {dup.sum()} duplicates (mean |diff| < 0.5) of {n - 1}')
    np.save(args.out.replace('.npy', '_diff.npy'), diff)

    if args.clock:
        x0, y0, x1, y1 = args.clock
        box = f[:, y0:y1, x0:x1]
        change = np.abs(np.diff(box, axis=0)).mean(axis=(1, 2))
        ticks = np.flatnonzero(change > 10 * np.median(change) + 2) + 1
        # merge the changes of consecutive frames (a digit fading in over two frames)
        ticks = ticks[np.insert(np.diff(ticks) > 2, 0, True)]
        np.save(args.out.replace('.npy', '_ticks.npy'), ticks)
        print('clock ticks (frame, video time s, frames since the previous tick, of them distinct):')
        for k, t in enumerate(ticks):
            gap = t - ticks[k - 1] if k else t
            distinct = (~dup[t - gap:t]).sum() if k else (~dup[:t]).sum()
            print(f'  {t:4d}  {t / args.fps:6.2f}  {gap:4d}  {distinct:4d}')


if __name__ == '__main__':
    main()
