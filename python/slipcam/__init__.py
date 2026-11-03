"""
slipcam: surface fault slip from video

  slipcam.decode_frames    video to grayscale frames, and their timing       (slipcam-decode)
  slipcam.track_offsets    dense offsets of the frames, framecor or pycuampcor (slipcam-track)
  slipcam.offset_map       maps of the offsets                               (slipcam-map)
  slipcam.region_series    time series of the offsets of boxes               (slipcam-regions)
  slipcam.slip_series      the slip of a box relative to another             (slipcam-slip)
  slipcam.stabilize        the frames warped onto a reference
  slipcam.auto_slip        the automatic pipeline                            (slipcam-auto)
  slipcam.framecor         the correlator (C++, CUDA and CPU backends)
"""

from ._version import __version__

__all__ = ['__version__']
