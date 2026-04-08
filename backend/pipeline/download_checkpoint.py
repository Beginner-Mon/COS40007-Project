"""
Download pretrained VideoPose3D checkpoint.

Usage:
    python -m pipeline.download_checkpoint

Downloads the ``pretrained_h36m_detectron_coco.bin`` weights into
``backend/pipeline/checkpoints/``.
"""

import os
import sys
import urllib.request
from pathlib import Path

CHECKPOINT_URL = (
    "https://dl.fbaipublicfiles.com/video-pose-3d/"
    "pretrained_h36m_detectron_coco.bin"
)
CHECKPOINT_DIR = Path(__file__).resolve().parent / "checkpoints"
CHECKPOINT_FILE = CHECKPOINT_DIR / "pretrained_h36m_detectron_coco.bin"
EXPECTED_SIZE_MB = 15  # approximate, just for sanity check


def download_checkpoint(force: bool = False) -> Path:
    """
    Download the pretrained VideoPose3D checkpoint if not already present.

    Parameters
    ----------
    force : bool
        Re-download even if the file already exists.

    Returns
    -------
    Path
        Absolute path to the downloaded checkpoint.
    """
    CHECKPOINT_DIR.mkdir(parents=True, exist_ok=True)

    if CHECKPOINT_FILE.exists() and not force:
        size_mb = CHECKPOINT_FILE.stat().st_size / (1024 * 1024)
        print(
            f"[checkpoint] Already exists: {CHECKPOINT_FILE} ({size_mb:.1f} MB)",
            flush=True,
        )
        return CHECKPOINT_FILE

    print(f"[checkpoint] Downloading from {CHECKPOINT_URL} ...", flush=True)

    def _progress(block_num, block_size, total_size):
        downloaded = block_num * block_size
        if total_size > 0:
            pct = downloaded / total_size * 100
            mb = downloaded / (1024 * 1024)
            total_mb = total_size / (1024 * 1024)
            sys.stdout.write(f"\r  {mb:.1f}/{total_mb:.1f} MB ({pct:.0f}%)")
            sys.stdout.flush()

    urllib.request.urlretrieve(CHECKPOINT_URL, str(CHECKPOINT_FILE), reporthook=_progress)
    print(flush=True)

    size_mb = CHECKPOINT_FILE.stat().st_size / (1024 * 1024)
    print(f"[checkpoint] Saved to {CHECKPOINT_FILE} ({size_mb:.1f} MB)", flush=True)
    return CHECKPOINT_FILE


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Download VideoPose3D pretrained weights.")
    parser.add_argument("--force", action="store_true", help="Re-download even if file exists.")
    args = parser.parse_args()
    download_checkpoint(force=args.force)
