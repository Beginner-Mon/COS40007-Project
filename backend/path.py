# path.py (đặt ở project root)

from pathlib import Path
from hydra.utils import get_original_cwd

def get_project_root() -> Path:
    """
    Return absolute path to project root,
    even when Hydra changes working directory.
    """
    return Path(get_original_cwd()).resolve()


# Project root
PROJECT_ROOT = get_project_root()

# Main folders
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = BACKEND_DIR / "dataset"


def show_paths():
    print("Project root :", PROJECT_ROOT)
    print("Backend      :", BACKEND_DIR)
    print("Frontend     :", FRONTEND_DIR)
    print("Dataset      :", DATA_DIR)


if __name__ == "__main__":
    show_paths()
