"""Bounded, verified decoding for rollout H.264 and public recovery AV1."""
from pathlib import Path
import operator
import shutil
import subprocess
import threading
import numpy as np
FFMPEG_DECODE_TIMEOUT_SECONDS = 120.0

class _VideoHandle:
    """Frame reader supporting ordinary MP4 and LeRobot's AV1 MP4.

    OpenCV is kept for H.264 rollout videos because it provides inexpensive
    random seeks.  The public LeRobot expert uses AV1; on systems where the
    OpenCV build cannot decode it, PyAV/libdav1d decodes that episode once and
    keeps uint8 frames in a bounded episode cache.  ``release`` drops all
    decoded frames when the parent episode is evicted.
    """

    def __init__(self, path, expected_frames=None, prefer_pyav=False):
        self.path = Path(path)
        self.expected_frames = expected_frames
        self._cv2 = None
        self._first = None
        self._frames = None
        if prefer_pyav:
            self._open_pyav()
            return
        try:
            import cv2
            capture = cv2.VideoCapture(str(self.path))
            if capture.isOpened():
                ok, first = capture.read()
                if ok:
                    self._cv2 = capture
                    self._first = first
                    if self.expected_frames is not None:
                        reported = int(round(capture.get(cv2.CAP_PROP_FRAME_COUNT)))
                        if reported and reported != self.expected_frames:
                            capture.release()
                            self._cv2 = None
                            self._first = None
                            raise ValueError(
                                f"video frame count {reported} disagrees with trajectory "
                                f"{self.expected_frames}: {self.path}"
                            )
                    return
                capture.release()
        except ImportError:
            pass
        self._open_pyav()

    def _open_pyav(self):
        try:
            import av
        except ImportError:
            self._open_imageio_ffmpeg()
            return
        try:
            with av.open(str(self.path)) as container:
                stream = container.streams.video[0]
                self._frames = [
                    frame.to_ndarray(format="bgr24")
                    for frame in container.decode(stream)
                ]
        except Exception as exc:
            try:
                self._open_imageio_ffmpeg()
                return
            except Exception as ffmpeg_exc:
                raise ValueError(
                    f"cannot decode video {self.path}: {exc}; "
                    f"imageio-ffmpeg fallback failed: {ffmpeg_exc}"
                ) from ffmpeg_exc
        if not self._frames:
            raise ValueError(f"video contains no decodable frames: {self.path}")
        if self.expected_frames is not None and len(self._frames) != self.expected_frames:
            raise ValueError(f"video frame count disagrees with trajectory: {self.path}")

    def _open_imageio_ffmpeg(self):
        """Decode without imageio's hard-coded ten-second metadata timeout.

        FFmpeg emits self-describing PPM frames, so this path needs neither
        FFprobe nor a guessed resolution.  Pixels are streamed one frame at a
        time and converted from RGB to the same BGR arrays as OpenCV/PyAV.
        A watchdog bounds the complete decode, including the first frame.
        """
        try:
            import imageio_ffmpeg
            executable = imageio_ffmpeg.get_ffmpeg_exe()
        except ImportError:
            executable = shutil.which("ffmpeg")
            if not executable:
                raise ValueError(
                    "cannot decode AV1; PyAV, imageio-ffmpeg, and ffmpeg are unavailable"
                )
        command = [
            executable,
            "-nostdin", "-hide_banner", "-loglevel", "error",
            "-threads", "1",
            "-i", str(self.path),
            "-map", "0:v:0",
            "-vsync", "0",
            "-filter_threads", "1",
            "-threads", "1",
            "-f", "image2pipe",
            "-c:v", "ppm",
            "-pix_fmt", "rgb24",
            "pipe:1",
        ]
        process = subprocess.Popen(
            command,
            stdin=subprocess.DEVNULL,
            stdout=subprocess.PIPE,
            stderr=subprocess.PIPE,
        )
        timed_out = threading.Event()
        error_tail = bytearray()

        def kill_after_timeout():
            if process.poll() is None:
                timed_out.set()
                process.kill()

        def drain_stderr():
            # Drain concurrently so diagnostic output cannot fill a pipe and
            # stall the decoder.  Retain only a bounded tail for error reports.
            while True:
                chunk = process.stderr.read(4096)
                if not chunk:
                    return
                error_tail.extend(chunk)
                del error_tail[:-8192]

        timer = threading.Timer(FFMPEG_DECODE_TIMEOUT_SECONDS, kill_after_timeout)
        timer.daemon = True
        stderr_thread = threading.Thread(target=drain_stderr, daemon=True)
        stderr_thread.start()
        timer.start()
        decoded = []
        error = None
        try:
            decoded = self._read_ppm_frames(process.stdout)
            process.wait()
        except Exception as exc:
            error = exc
        finally:
            timer.cancel()
            if process.poll() is None:
                process.kill()
            process.wait()
            stderr_thread.join()
            process.stdout.close()
            process.stderr.close()
        detail = bytes(error_tail).decode("utf-8", errors="replace").strip()
        if timed_out.is_set():
            raise ValueError(
                f"FFmpeg decoding exceeded {FFMPEG_DECODE_TIMEOUT_SECONDS:g}s "
                f"for {self.path}: {detail[-1000:]}"
            ) from error
        if process.returncode:
            raise ValueError(
                f"FFmpeg failed ({process.returncode}) for {self.path}: "
                f"{detail[-1000:]}"
            ) from error
        if error is not None:
            raise ValueError(f"cannot decode video {self.path}: {error}") from error
        if not decoded:
            raise ValueError(f"video contains no decodable frames: {self.path}")
        if self.expected_frames is not None and len(decoded) != self.expected_frames:
            raise ValueError(
                f"video frame count {len(decoded)} disagrees with trajectory "
                f"{self.expected_frames}: {self.path}"
            )
        self._frames = decoded

    def _read_ppm_frames(self, stream):
        """Read FFmpeg's P6 stream without buffering the full raw stdout.

        FFmpeg writes a three-line PPM header (magic, dimensions, maxval).
        We deliberately parse that output format instead of a general image
        file format.  Strict size limits reject malformed headers before a
        huge allocation; they do not resize or omit any training frame.
        """
        decoded = []
        total_bytes = 0
        max_frame_bytes = 64 * 1024 * 1024
        max_episode_bytes = 2 * 1024 * 1024 * 1024
        while True:
            magic = stream.readline(128)
            if not magic:
                return decoded
            if magic != b"P6\n":
                raise ValueError(f"invalid FFmpeg PPM magic: {magic!r}")
            dimensions = stream.readline(128)
            try:
                width, height = (int(value) for value in dimensions.split())
            except ValueError as exc:
                raise ValueError(f"invalid FFmpeg PPM dimensions: {dimensions!r}") from exc
            if stream.readline(128) != b"255\n":
                raise ValueError("unsupported FFmpeg PPM maxval")
            frame_bytes = width * height * 3
            if width <= 0 or height <= 0 or frame_bytes > max_frame_bytes:
                raise ValueError(f"invalid or excessive video dimensions: {width}x{height}")
            if self.expected_frames is not None and len(decoded) >= self.expected_frames:
                raise ValueError(
                    f"video has more frames than trajectory {self.expected_frames}"
                )
            total_bytes += frame_bytes
            if total_bytes > max_episode_bytes:
                raise ValueError("decoded episode exceeds the 2 GiB safety limit")
            pixels = stream.read(frame_bytes)
            if len(pixels) != frame_bytes:
                raise ValueError(
                    f"truncated FFmpeg frame {len(decoded)}: "
                    f"{len(pixels)} bytes, expected {frame_bytes}"
                )
            rgb = np.frombuffer(pixels, dtype=np.uint8).reshape(height, width, 3)
            decoded.append(rgb[:, :, ::-1].copy())

    def read(self, frame):
        if isinstance(frame, (bool, np.bool_)):
            raise ValueError("Video frame must be a nonnegative integer")
        try:
            frame = operator.index(frame)
        except TypeError as exc:
            raise ValueError("Video frame must be a nonnegative integer") from exc
        if frame < 0:
            raise ValueError("Video frame must be a nonnegative integer")
        if self._frames is not None:
            if frame >= len(self._frames):
                raise ValueError(f"missing video frame {frame} in {self.path}")
            return self._frames[frame]
        import cv2
        if frame == 0 and self._first is not None:
            value = self._first
            self._first = None
            return value
        self._cv2.set(cv2.CAP_PROP_POS_FRAMES, frame)
        ok, value = self._cv2.read()
        if not ok:
            raise ValueError(f"missing video frame {frame} in {self.path}")
        return value

    def release(self):
        if self._cv2 is not None:
            self._cv2.release()
            self._cv2 = None
        self._first = None
        self._frames = None
