"""
Video frame extraction utility.

Provides a general-purpose video loader that any converter can use
when the dataset's images need to be extracted from video files.

Supports: MP4, FLV (with FFmpeg transcode), AVI, MOV, MKV
"""

import os
import shutil
import subprocess
import uuid
from pathlib import Path
from typing import Optional

try:
    import cv2
    HAS_CV2 = True
    _OPENCV_HAS_FFMPEG = "FFMPEG:                      YES" in cv2.getBuildInformation()
except ImportError:
    HAS_CV2 = False
    _OPENCV_HAS_FFMPEG = False

VIDEO_EXTS = {".mp4", ".flv", ".avi", ".mov", ".mkv", ".wmv", ".m4v"}


class VideoLoader:
    """
    Load video files and extract frames by index.

    Usage:
        loader = VideoLoader("/path/to/videos")
        loader.load_all()                    # load all videos into memory
        frame = loader.get_frame("video1", 42)  # get frame 42 of video1
        path = loader.save_frame(frame, "/output", "video1", 42)  # save as jpg
    """

    def __init__(self, video_dir: str):
        self.video_dir = Path(video_dir)
        self.videos: dict[str, list] = {}  # key → list of frames
        self.video_info: dict[str, dict] = {}  # key → metadata

    def load_all(self, prefer_mp4: bool = True) -> dict:
        """
        Load all video files from the directory.

        Args:
            prefer_mp4: if both .flv and .mp4 exist for same stem, use .mp4

        Returns:
            dict of {video_key: frame_count}
        """
        if not HAS_CV2:
            raise RuntimeError(
                "OpenCV is required for video frame extraction. "
                "Install with: pip install opencv-python"
            )

        all_files = sorted(
            f for f in os.listdir(self.video_dir)
            if os.path.isfile(self.video_dir / f)
        )

        mp4_stems = set()
        if prefer_mp4:
            mp4_stems = {
                os.path.splitext(f)[0]
                for f in all_files
                if f.lower().endswith(".mp4")
            }

        loaded = {}
        for filename in all_files:
            ext = os.path.splitext(filename)[1].lower()
            if ext not in VIDEO_EXTS:
                continue

            key = os.path.splitext(filename)[0]

            # Skip FLV if MP4 exists
            if prefer_mp4 and ext == ".flv" and key in mp4_stems:
                continue

            path = str(self.video_dir / filename)
            frames, info = self._decode_video(path)

            if frames is not None:
                self.videos[key] = frames
                self.video_info[key] = info
                loaded[key] = len(frames)
                print(f"  Loaded {filename}: {len(frames)} frames "
                      f"({info.get('width')}x{info.get('height')} @ {info.get('fps')}fps)")
            else:
                print(f"  Skipping unreadable video: {filename}")

        return loaded

    def load_single(self, video_path: str) -> Optional[str]:
        """
        Load a single video file.

        Returns:
            video key if successful, None if failed
        """
        if not HAS_CV2:
            raise RuntimeError("OpenCV is required for video frame extraction.")

        path = Path(video_path)
        key = path.stem
        frames, info = self._decode_video(str(path))

        if frames is not None:
            self.videos[key] = frames
            self.video_info[key] = info
            return key
        return None

    def get_frame(self, video_key: str, frame_index: int):
        """Get a specific frame from a loaded video."""
        frames = self.videos.get(video_key)
        if frames is None:
            return None
        if frame_index < 0 or frame_index >= len(frames):
            return None
        return frames[frame_index]

    def get_frame_count(self, video_key: str) -> int:
        """Get the number of frames for a loaded video."""
        frames = self.videos.get(video_key)
        return len(frames) if frames else 0

    def get_video_keys(self) -> list[str]:
        """Get all loaded video keys."""
        return list(self.videos.keys())

    def save_frame(
        self,
        frame,
        output_dir: str,
        video_key: str,
        frame_index: int,
        ext: str = ".jpg",
    ) -> Optional[str]:
        """
        Save a frame as an image file with a unique name.

        Returns:
            filename (not full path) of the saved image, or None on failure
        """
        os.makedirs(output_dir, exist_ok=True)

        safe_key = "".join(
            ch if ch.isalnum() or ch in ("-", "_") else "_"
            for ch in video_key
        )
        filename = f"{safe_key}_frame_{frame_index:06d}_{uuid.uuid4().hex}{ext}"
        filepath = os.path.join(output_dir, filename)

        if not cv2.imwrite(filepath, frame):
            print(f"Failed to save frame: {filepath}")
            return None

        return filename

    def _decode_video(self, path: str) -> tuple:
        """
        Decode all frames from a video file.

        Returns:
            (list_of_frames, info_dict) or (None, None) on failure
        """
        cap = self._open_video(path)
        if cap is None or not cap.isOpened():
            if cap:
                cap.release()
            return None, None

        info = {
            "fps": cap.get(cv2.CAP_PROP_FPS),
            "width": int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)),
            "height": int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)),
            "total_frames": int(cap.get(cv2.CAP_PROP_FRAME_COUNT)),
        }

        frames = []
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            frames.append(frame)
        cap.release()

        if not frames:
            return None, None

        info["decoded_frames"] = len(frames)
        return frames, info

    def _open_video(self, path: str):
        """Open a video file with the best available backend."""
        lower = path.lower()
        cap = None

        # Try FFMPEG backend first
        if _OPENCV_HAS_FFMPEG:
            cap = cv2.VideoCapture(path, cv2.CAP_FFMPEG)
            if not cap.isOpened():
                cap.release()
                cap = cv2.VideoCapture(path)
        else:
            cap = cv2.VideoCapture(path)

        # If FLV fails, try transcoding to MP4
        if not cap.isOpened() and lower.endswith(".flv"):
            transcoded = self._transcode_flv(path)
            if transcoded:
                cap.release()
                cap = cv2.VideoCapture(transcoded)

        return cap

    def _transcode_flv(self, input_path: str) -> Optional[str]:
        """Transcode FLV to MP4 using FFmpeg."""
        ffmpeg_path = shutil.which("ffmpeg")
        if not ffmpeg_path:
            print("FFmpeg not found — cannot transcode FLV files.")
            return None

        output_path = f"{os.path.splitext(input_path)[0]}.mp4"
        if os.path.exists(output_path):
            return output_path

        result = subprocess.run(
            [ffmpeg_path, "-y", "-i", input_path,
             "-an", "-c:v", "libx264", "-pix_fmt", "yuv420p", output_path],
            capture_output=True, text=True,
        )
        if result.returncode != 0:
            return None
        return output_path
