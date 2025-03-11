import os
import json
import pandas as pd
import matplotlib.pyplot as plt


json_files = [
    "synthetic_ti_v1_res_ds_fid.json", "synthetic_ti_v1_res_ds.json",
    "synthetic_ti_v1_upscaled_res_ds_fid.json", "synthetic_ti_v1_upscaled_res_ds.json",
    "synthetic_ti_v2_res_ds_fid.json", "synthetic_ti_v2_res_ds.json",
    "synthetic_ti_v2_upscaled_res_ds_fid.json", "synthetic_ti_v2_upscaled_res_ds.json",
    "synthetic_v0_res_ds_fid.json", "synthetic_v0_res_ds.json",
    "synthetic_v0_upscaled_res_ds_fid.json", "synthetic_v0_upscaled_res_ds.json",
    "synthetic_v1_res_ds_fid.json", "synthetic_v1_res_ds.json",
    "synthetic_v1_upscaled_res_ds_fid.json", "synthetic_v1_upscaled_res_ds.json",
    "synthetic_v2_res_ds_fid.json", "synthetic_v2_res_ds.json",
    "synthetic_v2_upscaled_res_ds_fid.json", "synthetic_v2_upscaled_res_ds.json"
]

base_dir = "/Users/dinushijayasinghe/Desktop/R2/aerial-image-classification/similarity/"

combined_data = {}

for file in sorted(json_files):
    file_path = os.path.join(base_dir, file)
    if os.path.exists(file_path):
        with open(file_path, 'r') as f:
            data = json.load(f)
            base_name = file.replace("_fid", "")
            base_name = base_name.replace("_res", "")
            base_name = base_name.replace("_ds", "")
            base_name = base_name.replace(".json", "")
            base_name = base_name.replace("synthetic_", "")
            base_name = base_name.replace("upscaled", "up")
            
            if base_name not in combined_data:
                combined_data[base_name] = {"Dataset": base_name, "CMMD": "-", "FID": "-"}
            
            if "_fid" in file:
                fid_score = data.get("dataset_fid_score")
                if fid_score is not None:
                    combined_data[base_name]["FID"] = fid_score
            else:
                cmmd_score = data.get("dataset_cmmd_score")
                if cmmd_score is not None:
                    combined_data[base_name]["CMMD"] = cmmd_score

combined_df = pd.DataFrame(list(combined_data.values()))

print("CMMD and FID Scores Table:")
print(combined_df.to_string(index=False))

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))

# Plot CMMD on the left y-axis
color = "tab:blue"
ax1.set_xlabel("Dataset")
ax1.set_ylabel("CMMD", color=color)
ax1.plot(combined_df["Dataset"], combined_df["CMMD"].astype(float), color=color, marker="o", label="CMMD")
ax1.tick_params(axis="y", labelcolor=color)

# Create a second y-axis for FID
ax2 = ax1.twinx()
color = "tab:red"
ax2.set_ylabel("FID", color=color)
ax2.plot(combined_df["Dataset"], combined_df["FID"].astype(float), color=color, marker="x", label="FID")
ax2.tick_params(axis="y", labelcolor=color)

# Add title and adjust layout
plt.title("CMMD and FID Scores by Dataset")
fig.tight_layout()

# Rotate x-axis labels for better readability
plt.xticks(rotation=45, ha="right")

# Add legends
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

# Show the plot
plt.show()