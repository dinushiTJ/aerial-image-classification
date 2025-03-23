import os
import json
import pandas as pd
import matplotlib.pyplot as plt

json_files = [
    "synthetic_ti_v1_res_cls_fid.json", "synthetic_ti_v1_res_cls.json",
    "synthetic_ti_v1_upscaled_res_cls_fid.json", "synthetic_ti_v1_upscaled_res_cls.json",
    "synthetic_ti_v2_res_cls_fid.json", "synthetic_ti_v2_res_cls.json",
    "synthetic_ti_v2_upscaled_res_cls_fid.json", "synthetic_ti_v2_upscaled_res_cls.json",
    "synthetic_v0_res_cls_fid.json", "synthetic_v0_res_cls.json",
    "synthetic_v0_upscaled_res_cls_fid.json", "synthetic_v0_upscaled_res_cls.json",
    "synthetic_v1_res_cls_fid.json", "synthetic_v1_res_cls.json",
    "synthetic_v1_upscaled_res_cls_fid.json", "synthetic_v1_upscaled_res_cls.json",
    "synthetic_v2_res_cls_fid.json", "synthetic_v2_res_cls.json",
    "synthetic_v2_upscaled_res_cls_fid.json", "synthetic_v2_upscaled_res_cls.json"
]

base_dir = "/Users/dinushijayasinghe/Desktop/R2/aerial-image-classification/similarity/"

# Dictionary to store data for each class
class_data = {}

# Process files in pairs (CMMD and FID)
for i in range(0, len(json_files), 2):
    fid_file = json_files[i]
    cmmd_file = json_files[i + 1]

    fid_path = os.path.join(base_dir, fid_file)
    cmmd_path = os.path.join(base_dir, cmmd_file)

    if os.path.exists(fid_path) and os.path.exists(cmmd_path):
        with open(fid_path, 'r') as f:
            fid_data = json.load(f)
        with open(cmmd_path, 'r') as f:
            cmmd_data = json.load(f)

        dataset_name = fid_file.replace("synthetic_", "").replace("_res_cls_fid.json", "")
        dataset_name = dataset_name.replace("_upscaled", "_up")

        for class_name in fid_data:
            token = fid_data[class_name].get("token", class_name)  # Use class_name if token missing
            fid_score = fid_data[class_name].get("fid")
            cmmd_score = cmmd_data[class_name].get("cmmd")

            if token not in class_data:
                class_data[token] = []

            class_data[token].append({
                "Dataset": dataset_name,
                "CMMD": cmmd_score if cmmd_score is not None else float('nan'),
                "FID": fid_score if fid_score is not None else float('nan')
            })

# Process each class separately
for token, entries in class_data.items():
    df = pd.DataFrame(entries)

    # Drop missing values to avoid errors
    df.dropna(inplace=True)

    if df.empty:
        continue  # Skip if no valid data

    # Find lowest CMMD and FID scores
    lowest_cmmd = df.loc[df['CMMD'].idxmin()]
    lowest_fid = df.loc[df['FID'].idxmin()]

    print(f"Class: {token}")
    print(df.to_string(index=False))
    print("\nLowest Scores:")
    print(f"Lowest CMMD: {lowest_cmmd['CMMD']} (Dataset: {lowest_cmmd['Dataset']})")
    print(f"Lowest FID: {lowest_fid['FID']} (Dataset: {lowest_fid['Dataset']})")
    print("\n" + "=" * 50 + "\n")

    # Plotting
    fig, ax1 = plt.subplots(figsize=(12, 6))

    # Sort datasets properly
    df["Dataset"] = pd.Categorical(df["Dataset"], categories=df["Dataset"].unique(), ordered=True)
    df.sort_values("Dataset", inplace=True)

    color = "tab:blue"
    ax1.set_xlabel("Dataset")
    ax1.set_ylabel("CMMD", color=color)
    ax1.plot(df["Dataset"], df["CMMD"], color=color, marker="o", label="CMMD")
    ax1.tick_params(axis="y", labelcolor=color)

    # Second axis for FID
    ax2 = ax1.twinx()
    color = "tab:red"
    ax2.set_ylabel("FID", color=color)
    ax2.plot(df["Dataset"], df["FID"], color=color, marker="x", label="FID")
    ax2.tick_params(axis="y", labelcolor=color)

    # Formatting
    plt.title(f"CMMD and FID Scores by Dataset (Class: {token})")
    fig.tight_layout()
    plt.xticks(rotation=45, ha="right")

    ax1.legend(loc="upper left")
    ax2.legend(loc="upper right")

    plt.show()
