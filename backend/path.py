from pathlib import Path

# Absolute path to project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent

# Main folders
BACKEND_DIR = PROJECT_ROOT / "backend"
FRONTEND_DIR = PROJECT_ROOT / "frontend"
DATA_DIR = BACKEND_DIR / "dataset"


def show_paths():
    print("Project root :", PROJECT_ROOT)
    print("Backend      :", BACKEND_DIR)
    print("Frontend     :", FRONTEND_DIR)
    print("Theme2       :", DATA_DIR)

if __name__ == "__main__":
    show_paths()