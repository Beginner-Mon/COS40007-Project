import os
from path import DATA_DIR
from .base import ExcelReader
import pandas as pd


class DataLoader:
    """
    Loads and concatenates datasets according to dataset structure:
    - boning
    - slicing
    - optionally combine P1 and P2
    """

    def __init__(self, combine_persons: bool = True):
        self.combine_persons = combine_persons
        self.reader = ExcelReader()

    def _load_activity(self, person_dir: str, activity: str) -> pd.DataFrame:
        """
        Load and concatenate all Excel files for one activity (boning/slicing)
        under one person directory.
        """
        print(f"  Loading activity: {activity}...")
        
        activity_path = os.path.join(person_dir, activity)
        dfs = []

        for file in os.listdir(activity_path):
            if file.endswith(".xlsx"):
                file_path = os.path.join(activity_path, file)
                dfs.extend(self.reader.read_excel(file_path))

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
        Save loaded datasets to CSV files.
        """
        os.makedirs(output_dir, exist_ok=True)

        for key, value in data.items():
            if isinstance(value, dict):
                for sub_key, df in value.items():
                    path = os.path.join(output_dir, f"{key}_{sub_key}.csv")
                    df.to_csv(path, index=False)
            else:
                path = os.path.join(output_dir, f"{key}.csv")
                value.to_csv(path, index=False)

        print(f"CSV files saved to '{output_dir}/'")

if __name__ == "__main__":
    loader = DataLoader(combine_persons=False)
    data = loader.load()
    print(data["P1"]["boning"].head())  # Example to show loaded data
    print(data["P2"]["slicing"].head())  # Example to show loaded data
    loader.save_csv(data, output_dir="output_data")