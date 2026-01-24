from pathlib import Path
from .base import Dataset
from path import DATA_DIR
class SharpnessDataset(Dataset):
    LABEL_MAP = {
        "Boning": 0,
        "Slicing": 1
    }

    def __init__(self, excel_path: Path, action: str):
        super().__init__(excel_path)
        self.label = self.LABEL_MAP[action]

    def get_sample(self):
        data = self.load()
        return data, self.label

class CutDataset(Dataset):
    def __init__(self, excel_path: Path):
        super().__init__(excel_path)
        self.sharpness_label = self._parse_sharpness()

    def _parse_sharpness(self):
        # Example: MVN-S-Boning-63-001.xlsx
        name = self.excel_path.stem
        parts = name.split("-")

        sharpness = int(parts[-2])

        if sharpness > 84:
            return 0  # Sharp
        elif sharpness >= 70:
            return 1  # Medium
        else:
            return 2  # Blunt

    def get_sample(self):
        data = self.load()
        return data, self.sharpness_label

