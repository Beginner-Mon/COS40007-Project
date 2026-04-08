"""
Stage 1 — Video → 2D Pose Estimation using MediaPipe Pose.

Extracts 33 landmarks (x, y, z, visibility) per frame from a video file.
Supports both the legacy ``mp.solutions.pose`` API and the modern
``mediapipe.tasks.vision.PoseLandmarker`` API (mediapipe ≥ 0.10.21).
"""

import cv2
import numpy as np
import urllib.request
from pathlib import Path

# ---------------------------------------------------------------------------
# Auto-detect which MediaPipe API is available
# ---------------------------------------------------------------------------
_USE_TASKS_API: bool | None = None

def _detect_api():
    global _USE_TASKS_API
    if _USE_TASKS_API is not None:
        return
    try:
        import mediapipe as mp
        _ = mp.solutions.pose
        _USE_TASKS_API = False
        print("[video_to_2d] Using legacy MediaPipe solutions API.", flush=True)
    except AttributeError:
        _USE_TASKS_API = True
        print("[video_to_2d] Using modern MediaPipe Tasks API.", flush=True)


# ---------------------------------------------------------------------------
# Model asset download (Tasks API only)
# ---------------------------------------------------------------------------
_MODEL_DIR = Path(__file__).resolve().parent / "checkpoints"
_POSE_MODEL_PATH = _MODEL_DIR / "pose_landmarker_heavy.task"
_POSE_MODEL_URL = (
    "https://storage.googleapis.com/mediapipe-models/"
    "pose_landmarker/pose_landmarker_heavy/float16/latest/"
    "pose_landmarker_heavy.task"
)


def _ensure_pose_model() -> Path:
    """Download the PoseLandmarker model if not already present."""
    if _POSE_MODEL_PATH.exists():
        return _POSE_MODEL_PATH

    _MODEL_DIR.mkdir(parents=True, exist_ok=True)
    print(f"[video_to_2d] Downloading PoseLandmarker model...", flush=True)

    import sys
    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded / total_size * 100
            mb = downloaded / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f} MB ({pct:.0f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(_POSE_MODEL_URL, str(_POSE_MODEL_PATH), reporthook=_progress)
    print(f"\n[video_to_2d] Model saved to {_POSE_MODEL_PATH}", flush=True)
    return _POSE_MODEL_PATH


# ---------------------------------------------------------------------------
# Extraction — Tasks API
# ---------------------------------------------------------------------------
def _extract_tasks_api(
    video_path: str,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> tuple[np.ndarray, dict]:
    """Use ``mediapipe.tasks.vision.PoseLandmarker``."""
    import mediapipe as mp
    from mediapipe.tasks import python as mp_python
    from mediapipe.tasks.python import vision as mp_vision

    model_path = _ensure_pose_model()

    options = mp_vision.PoseLandmarkerOptions(
        base_options=mp_python.BaseOptions(model_asset_path=str(model_path)),
        running_mode=mp_vision.RunningMode.VIDEO,
        num_poses=1,
        min_pose_detection_confidence=min_detection_confidence,
        min_pose_presence_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    landmarker = mp_vision.PoseLandmarker.create_from_options(options)

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    all_keypoints: list[np.ndarray] = []
    detected_count = 0
    frame_idx = 0

    print(f"[video_to_2d] Processing {total_frames} frames...", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb)

        timestamp_ms = int(frame_idx * 1000 / max(fps, 1))
        result = landmarker.detect_for_video(mp_image, timestamp_ms)

        if result.pose_landmarks and len(result.pose_landmarks) > 0:
            landmarks = result.pose_landmarks[0]
            kp = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
                dtype=np.float32,
            )
            all_keypoints.append(kp)
            detected_count += 1
        else:
            all_keypoints.append(np.full((33, 4), np.nan, dtype=np.float32))

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{total_frames}", flush=True)

    cap.release()
    landmarker.close()

    keypoints_2d = np.stack(all_keypoints, axis=0)
    metadata = {
        "fps": fps, "width": width, "height": height,
        "total_frames": frame_idx, "detected_frames": detected_count,
    }
    return keypoints_2d, metadata


# ---------------------------------------------------------------------------
# Extraction — Legacy solutions API
# ---------------------------------------------------------------------------
def _extract_legacy_api(
    video_path: str,
    model_complexity: int,
    min_detection_confidence: float,
    min_tracking_confidence: float,
) -> tuple[np.ndarray, dict]:
    """Use ``mp.solutions.pose.Pose``."""
    import mediapipe as mp

    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Cannot open video: {video_path}")

    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))

    mp_pose = mp.solutions.pose
    pose = mp_pose.Pose(
        static_image_mode=False,
        model_complexity=model_complexity,
        min_detection_confidence=min_detection_confidence,
        min_tracking_confidence=min_tracking_confidence,
    )

    all_keypoints: list[np.ndarray] = []
    detected_count = 0
    frame_idx = 0

    print(f"[video_to_2d] Processing {total_frames} frames...", flush=True)

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        results = pose.process(rgb)

        if results.pose_landmarks:
            landmarks = results.pose_landmarks.landmark
            kp = np.array(
                [[lm.x, lm.y, lm.z, lm.visibility] for lm in landmarks],
                dtype=np.float32,
            )
            all_keypoints.append(kp)
            detected_count += 1
        else:
            all_keypoints.append(np.full((33, 4), np.nan, dtype=np.float32))

        frame_idx += 1
        if frame_idx % 100 == 0:
            print(f"  frame {frame_idx}/{total_frames}", flush=True)

    cap.release()
    pose.close()

    keypoints_2d = np.stack(all_keypoints, axis=0)
    metadata = {
        "fps": fps, "width": width, "height": height,
        "total_frames": frame_idx, "detected_frames": detected_count,
    }
    return keypoints_2d, metadata


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------
def extract_2d_poses(
    video_path: str,
    model_complexity: int = 2,
    min_detection_confidence: float = 0.5,
    min_tracking_confidence: float = 0.5,
) -> tuple[np.ndarray, dict]:
    """
    Run MediaPipe Pose on each frame of a video.

    Parameters
    ----------
    video_path : str
        Path to the input video file (.mp4, .avi, .mov, etc.).
    model_complexity : int
        MediaPipe model complexity (0=lite, 1=full, 2=heavy).
        Only used with the legacy solutions API.
    min_detection_confidence : float
        Minimum confidence for person detection.
    min_tracking_confidence : float
        Minimum confidence for landmark tracking.

    Returns
    -------
    keypoints_2d : np.ndarray
        Shape ``(num_frames, 33, 4)`` — 33 landmarks × (x, y, z, visibility).
        Frames where no pose was detected are filled with NaN.
    metadata : dict
        ``{"fps": float, "width": int, "height": int,
           "total_frames": int, "detected_frames": int}``
    """
    _detect_api()

    if _USE_TASKS_API:
        keypoints_2d, metadata = _extract_tasks_api(
            video_path, min_detection_confidence, min_tracking_confidence,
        )
    else:
        keypoints_2d, metadata = _extract_legacy_api(
            video_path, model_complexity,
            min_detection_confidence, min_tracking_confidence,
        )

    print(
        f"[video_to_2d] Done. {metadata['detected_frames']}/{metadata['total_frames']} "
        f"frames detected ({metadata['detected_frames'] / max(metadata['total_frames'], 1) * 100:.1f}%)",
        flush=True,
    )
    return keypoints_2d, metadata
