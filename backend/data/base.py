import os
import re
import pandas as pd
from non_hydra_path import DATA_DIR
from pathlib import Path

class ExcelReader:
    """
    Read Segment Velocity and Segment Acceleration sheets
    and inject metadata labels directly.
    """

    def _load_markers(self, file_path: Path) -> list[tuple[int, int, str]]:
        markers = pd.read_excel(file_path, sheet_name="Markers")
        if markers.empty:
            return []

        frame_col = None
        label_col = None
        for col in markers.columns:
            col_str = str(col).strip().lower()
            if frame_col is None and "frame" in col_str:
                frame_col = col
            if label_col is None and "label" in col_str:
                label_col = col

        if frame_col is None or label_col is None:
            raise ValueError("Marker sheet must contain Frame and Label columns")

        ranges: list[tuple[int, int, str]] = []
        for _, row in markers.iterrows():
            frame_val = row[frame_col]
            label_val = row[label_col]
            if pd.isna(frame_val) or pd.isna(label_val):
                continue

            nums = re.findall(r"\d+", str(frame_val))
            if not nums:
                continue
            start = int(nums[0])
            end = int(nums[1]) if len(nums) > 1 else start
            label = str(label_val).strip()
            ranges.append((start, end, label))

        return ranges

    SHEETS = ["Segment Velocity", "Segment Acceleration"]

   

    def _parse_metadata(self, file_path: Path) -> dict:
        file_path = Path(file_path)
        file_path_str = str(file_path)

        # Person ID
        if "P1" in file_path_str:
            person_id = "P1"
        elif "P2" in file_path_str:
            person_id = "P2"
        else:
            raise ValueError("Person ID not found in path")

        # Activity type
        if "boning" in file_path_str.lower():
            activity_type = "boning"
        elif "slicing" in file_path_str.lower():
            activity_type = "slicing"
        else:
            raise ValueError("Activity type not found in path")

        # Filename
        filename = file_path.name
        match = re.search(r"-(\d{2,3})-", filename)
        if not match:
            raise ValueError(f"Sharpness score not found in {filename}")

        sharpness_score = int(match.group(1))

        return {
            "person_id": person_id,
            "activity_type": activity_type,
            "knife_sharpness_score": sharpness_score,
        }


    def read_excel(self, file_path: str) -> list[pd.DataFrame]:
        """
        Read required sheets and return labeled DataFrames.
        """
        metadata = self._parse_metadata(file_path)
        marker_ranges = self._load_markers(file_path)
        dfs = []

        for sheet in self.SHEETS:
            df = pd.read_excel(file_path, sheet_name=sheet)

            if "Frame" not in df.columns:
                raise ValueError(f"Frame column not found in sheet '{sheet}'")

            if "Marker" in df.columns:
                df = df.drop(columns=["Marker"])
            if "Label" in df.columns:
                df = df.drop(columns=["Label"])

            df["Label"] = pd.NA
            frame_series = pd.to_numeric(df["Frame"], errors="coerce")
            for start, end, label in marker_ranges:
                df.loc[(frame_series >= start) & (frame_series <= end), "Label"] = label

            # Inject labels
            for k, v in metadata.items():
                df[k] = v

            df["sensor_type"] = sheet  
            df["video_id"] = Path(file_path).stem
            dfs.append(df)

        return dfs

if __name__ == "__main__":
    reader = ExcelReader()
    data_frames = reader.read_excel( DATA_DIR / "P1/Slicing/MVN-J-Slicing-87-001.xlsx")
    print(data_frames[1].head())  # Example to show data read
    print(data_frames[1]["Label"].value_counts())
