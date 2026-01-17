"""
Extract 3D keypoint data from Excel file and save as JSON
"""

import pandas as pd
import json
import sys

# Read the Excel file
excel_path = r"c:\Users\laptop HP\OneDrive - Swinburne University\Swinburne documents\COS40007 AI for Engineering\Project\P1\P1\Boning\MVN-J-Boning-64-002.xlsx"
sheet_name = "Segment Position"

# Load the data
df = pd.read_excel(excel_path, sheet_name=sheet_name)

print("Excel file loaded successfully!")
print(f"Shape: {df.shape}")
print(f"Columns: {df.columns.tolist()}")
print(f"\nFirst few rows:")
print(df.head())

# Joint names in order
joints = [
    "Pelvis", "L5", "L3", "T12", "T8", "Neck", "Head",
    "Right Shoulder", "Right Upper Arm", "Right Forearm", "Right Hand",
    "Left Shoulder", "Left Upper Arm", "Left Forearm", "Left Hand",
    "Right Upper Leg", "Right Lower Leg", "Right Foot", "Right Toe",
    "Left Upper Leg", "Left Lower Leg", "Left Foot", "Left Toe"
]

# Verify we have all joint columns
print(f"\n\nVerifying joint columns...")
for joint in joints:
    x_col = f"{joint} x"
    y_col = f"{joint} y"
    z_col = f"{joint} z"
    if x_col in df.columns and y_col in df.columns and z_col in df.columns:
        print(f"✓ {joint}")
    else:
        print(f"✗ {joint} - Missing columns!")

# Create the output format
output = {
    "joints": joints,
    "fps": 60,
    "frames": []
}

# Extract frame data
for idx, row in df.iterrows():
    frame_num = int(row['Frame'])
    frame_data = {
        "frame": frame_num,
        "keypoints": []
    }
    
    # Extract x, y, z for each joint
    for joint in joints:
        x = float(row[f"{joint} x"])
        y = float(row[f"{joint} y"])
        z = float(row[f"{joint} z"])
        frame_data["keypoints"].append([x, y, z])
    
    output["frames"].append(frame_data)

print(f"\n\nExtracted {len(output['frames'])} frames")
print(f"Each frame has {len(output['joints'])} joints")

# Save to JSON
output_path = r"c:\Users\laptop HP\OneDrive - Swinburne University\Swinburne documents\COS40007 AI for Engineering\Project\frontend\public\keypoints_data.json"
with open(output_path, 'w') as f:
    json.dump(output, f)

print(f"\n✓ Data saved to: {output_path}")
print(f"File size: {len(json.dumps(output)) / 1024 / 1024:.2f} MB")
