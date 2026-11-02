"""Keep Gym's per-episode video interface while encoding frames immediately."""
from pathlib import Path
import gc

import gymnasium as gym
import imageio_ffmpeg
import numpy as np


class StreamingRecordVideo(gym.wrappers.RecordVideo):
    """Record RGB episodes without keeping hundreds of full frames in memory."""

    def __init__(self, *args, **kwargs):
        self._stream = None
        self._stream_path = None
        self._frame_count = 0
        super().__init__(*args, **kwargs)

    def start_recording(self, video_name):
        super().start_recording(video_name)
        self._frame_count = 0

    def _capture_frame(self):
        if not self.recording:
            raise RuntimeError('Capture requested before recording started.')
        frame = self.env.render()
        if not isinstance(frame, np.ndarray) or frame.ndim != 3 or frame.shape[2] != 3:
            raise RuntimeError('Streaming recording requires an RGB numpy frame.')
        if frame.dtype != np.uint8:
            raise RuntimeError('Streaming recording requires uint8 frames.')
        if self._stream is None:
            self._stream_path = Path(self.video_folder) / f'{self._video_name}.partial.mp4'
            self._stream = imageio_ffmpeg.write_frames(
                str(self._stream_path), (frame.shape[1], frame.shape[0]),
                fps=self.frames_per_sec, codec='libx264', pix_fmt_in='rgb24',
                pix_fmt_out='yuv420p', quality=5, macro_block_size=1,
                output_params=['-threads', '2', '-movflags', '+faststart'])
            self._stream.send(None)
        self._stream.send(np.ascontiguousarray(frame))
        self._frame_count += 1
        if self._frame_count > self.video_length:
            self.stop_recording()

    def stop_recording(self):
        if not self.recording:
            raise RuntimeError('Stop requested before recording started.')
        try:
            if self._stream is not None:
                self._stream.close()
                if not self._stream_path.exists() or self._stream_path.stat().st_size == 0:
                    raise RuntimeError('Video encoder did not produce a file.')
                self._stream_path.replace(Path(self.video_folder) / f'{self._video_name}.mp4')
        finally:
            self._stream = None
            self._stream_path = None
            self._frame_count = 0
            self.recorded_frames = []
            self.recording = False
            self._video_name = None
        if self.gc_trigger and self.gc_trigger(self.episode_id):
            gc.collect()
