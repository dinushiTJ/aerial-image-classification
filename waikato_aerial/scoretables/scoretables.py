import os
import json
import pandas as pd

import matplotlib
matplotlib.use('Agg')
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
    "synthetic_v2_upscaled_res_ds_fid.json", "synthetic_v2_upscaled_res_ds.json",
    # "synthetic_best_cmmd_res_ds.json", "synthetic_best_cmmd_res_ds_fid.json",
    # "synthetic_best_fid_res_ds.json", "synthetic_best_fid_res_ds_fid.json"
]

base_dir = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/similarity/res"
save_dir = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/scoretables/plots_ds"
suffix = ""

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

            if base_name == "best":
                base_name = "best_fid"
            
            if base_name not in combined_data:
                combined_data[base_name] = {"Dataset": base_name, "CMMD": "-", "FID": "-"}
            
            if file.endswith("_fid.json"):
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

combined_df.to_csv("v2_ds_scores{suffix}.csv")
combined_df.to_json("v2_ds_scores{suffix}.json")

# Find lowest CMMD and FID scores
lowest_cmmd = combined_df.loc[combined_df['CMMD'].idxmin()]
lowest_fid = combined_df.loc[combined_df['FID'].idxmin()]
print("\nLowest Scores:")
print(f"Lowest CMMD: {lowest_cmmd['CMMD']} (Dataset: {lowest_cmmd['Dataset']})")
print(f"Lowest FID: {lowest_fid['FID']} (Dataset: {lowest_fid['Dataset']})")

# Plotting
fig, ax1 = plt.subplots(figsize=(12, 6))
ax1.grid(True, linestyle='--', alpha=0.5)

# Plot CMMD on the left y-axis
color = "#d4d414"
ax1.set_xlabel("Dataset")
ax1.set_ylabel("CMMD")
ax1.plot(combined_df["Dataset"], combined_df["CMMD"].astype(float), color=color, marker="o", label="CMMD")
ax1.tick_params(axis="y")

# Add horizontal line for lowest CMMD
min_cmmd = combined_df["CMMD"].astype(float).min()
ax1.axhline(min_cmmd, color="black", linestyle="--", alpha=0.3, label="Lowest CMMD")

# Create a second y-axis for FID
ax2 = ax1.twinx()
ax2.grid(False)  # grid already added to ax1
color = "#10aabb"
ax2.set_ylabel("FID")
ax2.plot(combined_df["Dataset"], combined_df["FID"].astype(float), color=color, marker="x", label="FID")
ax2.tick_params(axis="y")

# Add horizontal line for lowest FID
min_fid = combined_df["FID"].astype(float).min()
ax2.axhline(min_fid, color="black", linestyle="--", alpha=0.3, label="Lowest FID")

# Add title and adjust layout
plt.title("CMMD and FID Scores by Dataset")
fig.tight_layout()
plt.xticks(rotation=45, ha="right")

# Add legends
ax1.legend(loc="upper left")
ax2.legend(loc="upper right")

plt.savefig(f"{save_dir}/v2_dataset_sm{suffix}.svg", format="svg")
plt.close()
print(f"Saved to {save_dir}/v2_dataset_sm{suffix}.svg")
