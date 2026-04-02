import pandas as pd

# Check what's actually in the output CSV for P2_slicing
df = pd.read_csv("output_data/P2_slicing.csv")
out = []

out.append(f"Total shape: {df.shape}")
out.append(f"sensor_type values: {df['sensor_type'].value_counts().to_dict()}")
out.append(f"video_id values: {df['video_id'].value_counts().to_dict()}")
out.append("")

# Cross-tab: video_id x sensor_type
ct = pd.crosstab(df['video_id'], df['sensor_type'])
out.append(f"Cross-tab (video_id x sensor_type):")
out.append(ct.to_string())

with open("_check_result.txt", "w") as f:
    f.write("\n".join(out))
print("Done")
