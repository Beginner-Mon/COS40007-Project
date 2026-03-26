import os
import argparse
import re
from non_hydra_path import DATA_DIR
from .base import ExcelReader
from .preprocessing import FEATURE_COLS
import pandas as pd


class DataLoader:
    """
    Loads and concatenates datasets according to dataset structure:
    - boning
    - slicing
    - optionally combine P1 and P2

    Preprocessing/quality checks done at loader stage:
    - enforce a consistent Label column (drop Marker),
    - validate expected motion feature columns,
    - concatenate per-file/per-sheet DataFrames,
    - export CSVs using default or sheet-suffixed naming rules.
    """

    def __init__(self, combine_persons: bool = True, sheets_to_process: list[str] | None = None):
        self.combine_persons = combine_persons
        self.reader = ExcelReader(sheets_to_process=sheets_to_process)

    @staticmethod
    def _normalize_sheet_for_suffix(sheet_name: str) -> str:
        cleaned = re.sub(r"[^a-zA-Z0-9]+", "_", sheet_name.strip().lower())
        return cleaned.strip("_") or "sheet"

    def _is_default_sheet_mode(self) -> bool:
        return set(self.reader.sheets_to_process) == set(ExcelReader.DEFAULT_SHEETS)

    @staticmethod
    def _normalize_label_columns(df: pd.DataFrame) -> pd.DataFrame:
        """
        Ensure a consistent label column:
        - Always drop Marker if present.
        - Require Label to exist (base.py should populate it).
        """
        if "Marker" in df.columns:
            df = df.drop(columns=["Marker"])

        if "Label" not in df.columns:
            raise ValueError("Missing Label column in dataset (expected from base.py)")

        return df

    @staticmethod
    def _validate_feature_columns(df: pd.DataFrame) -> None:
        missing = [c for c in FEATURE_COLS if c not in df.columns]
        if missing:
            missing_str = ", ".join(missing)
            raise ValueError(f"Missing feature columns in dataset: {missing_str}")

    def _load_activity(self, person_dir: str, activity: str) -> pd.DataFrame:
        """
        Load and concatenate all Excel files for one activity (boning/slicing)
        under one person directory.

        Per DataFrame, this method applies loader-side preprocessing checks:
        - normalize label-related columns,
        - enforce feature schema before concatenation.
        """
        print(f"  Loading activity: {activity}...")
        
        activity_path = os.path.join(person_dir, activity)
        dfs = []

        for file in os.listdir(activity_path):
            if file.endswith(".xlsx"):
                file_path = os.path.join(activity_path, file)
                for df in self.reader.read_excel(file_path):
                    df = self._normalize_label_columns(df)
                    self._validate_feature_columns(df)
                    dfs.append(df)

        return pd.concat(dfs, ignore_index=True)

    def load(self):
        """
        Main entry point.
        Returns:
          - if combine_persons=True:
              {"boning": df, "slicing": df}
          - else:
              {
                "P1": {"boning": df, "slicing": df},
                "P2": {"boning": df, "slicing": df}
              }
        """
        result = {}

        persons = ["P1", "P2"]
        print("Loading datasets...")
        # Combine persons
        if self.combine_persons:
            boning_dfs = []
            slicing_dfs = []

            for p in persons:
                print(f"Loading data for {p}...")
                person_dir = os.path.join(DATA_DIR, p)

                boning_dfs.append(self._load_activity(person_dir, "Boning"))
                slicing_dfs.append(self._load_activity(person_dir, "Slicing"))

            result["boning"] = pd.concat(boning_dfs, ignore_index=True)
            result["slicing"] = pd.concat(slicing_dfs, ignore_index=True)
        # Separate persons
        else:
            for p in persons:
                print(f"Loading data for {p}...")
                person_dir = os.path.join(DATA_DIR, p)
                result[p] = {
                    "boning": self._load_activity(person_dir, "Boning"),
                    "slicing": self._load_activity(person_dir, "Slicing"),
                }

        print("Datasets loaded.")
        return result
    
    def save_csv(self, data: dict, output_dir: str = "output"):
        """
        Save preprocessed datasets to CSV files.

        Naming behavior:
        - default sheets mode -> unsuffixed names (e.g., P1_boning.csv),
        - custom sheets mode -> one file per sheet with sheet suffix.
        """
        os.makedirs(output_dir, exist_ok=True)

        default_mode = self._is_default_sheet_mode()

        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, df in value.items():
                    if default_mode:
                        path = os.path.join(output_dir, f"{key}_{sub_key}.csv")
                        df.to_csv(path, index=False)
                    else:
                        for sheet in self.reader.sheets_to_process:
                            sheet_df = df[df["sensor_type"] == sheet]
                            suffix = self._normalize_sheet_for_suffix(sheet)
                            path = os.path.join(output_dir, f"{key}_{sub_key}_{suffix}.csv")
                            sheet_df.to_csv(path, index=False)
            else:
                if default_mode:
                    path = os.path.join(output_dir, f"{key}.csv")
                    value.to_csv(path, index=False)
                else:
                    for sheet in self.reader.sheets_to_process:
                        sheet_df = value[value["sensor_type"] == sheet]
                        suffix = self._normalize_sheet_for_suffix(sheet)
                        path = os.path.join(output_dir, f"{key}_{suffix}.csv")
                        sheet_df.to_csv(path, index=False)

        print(f"CSV files saved to '{output_dir}/'")


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Load Excel files and export processed CSVs for motion activity data."
    )
    parser.add_argument(
        "--combine-persons",
        action="store_true",
        help="Combine P1 and P2 into single boning/slicing outputs.",
    )
    parser.add_argument(
        "--sheets",
        nargs="+",
        default=None,
        help=(
            "Process only these sheet names (exclusive override). "
            "If omitted, defaults to Segment Velocity + Segment Acceleration."
        ),
    )
    parser.add_argument(
        "--output-dir",
        default="output_data",
        help="Output folder for generated CSV files.",
    )
    return parser.parse_args()

if __name__ == "__main__":
    args = _parse_args()
    loader = DataLoader(combine_persons=args.combine_persons, sheets_to_process=args.sheets)
    print(f"Effective sheets: {loader.reader.sheets_to_process}")
    data = loader.load()
    if not args.combine_persons:
        print(data["P1"]["boning"].head())  # Example to show loaded data
        print(data["P2"]["slicing"].head())  # Example to show loaded data
    else:
        print(data["boning"].head())
        print(data["slicing"].head())
    loader.save_csv(data, output_dir=args.output_dir)
