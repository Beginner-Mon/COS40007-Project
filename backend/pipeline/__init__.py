"""
Video-to-Prediction Pipeline.

End-to-end inference: Video → MediaPipe 2D → VideoPose3D 3D → Feature Engineering → BiLSTM Prediction.

Usage (CLI):
    python -m pipeline.video_inference <video_path> [--checkpoint <path>] [--output-json <path>]

Usage (Python):
    from pipeline.video_inference import predict_from_video
    result = predict_from_video("path/to/video.mp4")
"""


def predict_from_video(*args, **kwargs):
    """Lazy wrapper — imports are deferred until first call."""
    from .video_inference import predict_from_video as _impl
    return _impl(*args, **kwargs)


__all__ = ["predict_from_video"]
