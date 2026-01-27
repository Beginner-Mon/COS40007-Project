import os
import re
import pandas as pd
from path import DATA_DIR
from pathlib import Path

class ExcelReader:
    """
    Read Segment Velocity and Segment Acceleration sheets
    and inject metadata labels directly.
    """

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
        dfs = []

        for sheet in self.SHEETS:
            df = pd.read_excel(file_path, sheet_name=sheet)

            # Inject labels
            for k, v in metadata.items():
                df[k] = v

            df["sensor_type"] = sheet  
            df["video_id"] = Path(file_path).stem
            dfs.append(df)

        return dfs

if __name__ == "__main__":
    reader = ExcelReader()
    data_frames = reader.read_excel( DATA_DIR / "P1/Boning/MVN-J-Boning-64-001.xlsx")
    print(data_frames[1].head())  # Example to show data read