from pathlib import Path
import pandas as pd
import numpy as np

class Dataset:
    def __init__(self, excel_path: Path, outlier_method: str = "zscore"):
        self.excel_path = excel_path
        self.outlier_method = outlier_method

        self.velocity_df = None
        self.accel_df = None

    # ---------- Public API ----------
    def load(self):
        self._load_sheets()
        self._basic_clean()
        self._report_quality()
        return self.to_tensor()

    # ---------- Internal ----------
    def _load_sheets(self):
        xls = pd.ExcelFile(self.excel_path)

        if "Segment Velocity" not in xls.sheet_names:
            raise ValueError("Missing sheet: Segment Velocity")

        if "Segment Acceleration" not in xls.sheet_names:
            raise ValueError("Missing sheet: Segment Acceleration")

        self.velocity_df = pd.read_excel(
            self.excel_path, sheet_name="Segment Velocity"
        )

        self.accel_df = pd.read_excel(
            self.excel_path, sheet_name="Segment Acceleration"
        )

    def _basic_clean(self):
        for df in [self.velocity_df, self.accel_df]:
            # Sort by Frame
            df.sort_values("Frame", inplace=True)

            # Drop duplicate frames
            df.drop_duplicates(subset="Frame", inplace=True)

            # Fill missing values
            df.interpolate(inplace=True)
            df.ffill(inplace=True)
            df.bfill(inplace=True)

    def _report_quality(self):
        for name, df in [
            ("Velocity", self.velocity_df),
            ("Acceleration", self.accel_df),
        ]:
            nulls = df.isnull().sum().sum()
            if nulls > 0:
                print(f"[WARN] {name}: {nulls} null values")

            self._report_outliers(df, name)

    def _report_outliers(self, df, name):
        numeric = df.drop(columns=["Frame"], errors="ignore")

        if self.outlier_method == "zscore":
            z = (numeric - numeric.mean()) / numeric.std()
            outliers = (np.abs(z) > 3).sum().sum()

        elif self.outlier_method == "iqr":
            q1 = numeric.quantile(0.25)
            q3 = numeric.quantile(0.75)
            iqr = q3 - q1
            outliers = ((numeric < (q1 - 1.5 * iqr)) |
                        (numeric > (q3 + 1.5 * iqr))).sum().sum()
        else:
            outliers = 0

        if outliers > 0:
            print(f"[INFO] {name}: {outliers} outliers detected")

    def to_tensor(self):
        # Drop Frame column
        v = self.velocity_df.drop(columns=["Frame"], errors="ignore").to_numpy()
        a = self.accel_df.drop(columns=["Frame"], errors="ignore").to_numpy()

        return {
            "velocity": v,        # shape (T, Fv)
            "acceleration": a     # shape (T, Fa)
        }
