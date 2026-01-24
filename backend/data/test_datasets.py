from path import DATA_DIR
from pathlib import Path
from .datasets import SharpnessDataset, CutDataset
# CHANGE THIS if needed
TEST_FILE = (
    DATA_DIR
    / "P2"
    / "Boning"
    / "MVN-S-Boning-63-001.xlsx"
)

assert TEST_FILE.exists(), "Test Excel file not found"


print("=== CutDataset (one file) ===")

cut_ds = CutDataset(TEST_FILE)
X_cut, y_cut = cut_ds.get_sample()

print("File:", TEST_FILE.name)
print("Sharpness label:", y_cut)  # 0 / 1 / 2

print("Velocity shape:", X_cut["velocity"].shape)
print("Acceleration shape:", X_cut["acceleration"].shape)


print("\n=== SharpnessDataset (one file) ===")

# Label comes from folder name
action = TEST_FILE.parent.name  # Boning / Slicing

sharp_ds = SharpnessDataset(
    excel_path=TEST_FILE,
    action=action
)

X_sharp, y_sharp = sharp_ds.get_sample()

print("File:", TEST_FILE.name)
print("Action:", action)
print("Action label:", y_sharp)  # 0 or 1

print("Velocity shape:", X_sharp["velocity"].shape)
print("Acceleration shape:", X_sharp["acceleration"].shape)

