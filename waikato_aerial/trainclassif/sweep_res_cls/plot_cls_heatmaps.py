import json
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# --- CONFIG ---
BASE_DIR = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/trainclassif/sweep_res_cls"
INPUT_JSON = f"{BASE_DIR}/run_summary_cls.json"
OUTPUT_DIR = f"{BASE_DIR}/results/per_class_heatmaps"
MODELS = {
    "efficientnet": "EfficientNet B2",
    "resnet50": "ResNet 50",
    "vit": "ViT B 16"
}
TRAINING_MODES = {
    "tl": "Transfer Learning",
    "sft": "Partial Fine-tuning",
    "fft": "Full Fine-tuning"
}
DATASET_ORDER = ["real", "p10", "p25", "p50", "p75", "p100", "p125", "p150"]

# --- LOAD DATA ---
os.makedirs(OUTPUT_DIR, exist_ok=True)
with open(INPUT_JSON, "r") as f:
    data = json.load(f)

class_names = list(data.keys())

# --- GENERATE HEATMAPS ---
for model_key, model_label in MODELS.items():
    # COMBINED PLOT
    fig, axes = plt.subplots(1, 3, figsize=(18, len(class_names) * 0.5 + 3), sharey=True)
    for idx, (mode_key, mode_label) in enumerate(TRAINING_MODES.items()):
        heatmap_data = np.full((len(class_names), len(DATASET_ORDER)), np.nan)

        for row_idx, cls in enumerate(class_names):
            for col_idx, ds in enumerate(DATASET_ORDER):
                try:
                    acc_list = data[cls][model_key][mode_key][ds]["accuracy"]
                    heatmap_data[row_idx][col_idx] = np.mean(acc_list)
                except KeyError:
                    continue

        ax = axes[idx]
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            xticklabels=DATASET_ORDER,
            yticklabels=class_names,
            cmap="YlGnBu",
            cbar=(idx == 2),
            ax=ax,
            linewidths=0.5,
            linecolor='gray',
            annot_kws={"fontsize": 8}
        )
        ax.set_title(mode_label, fontsize=13)
        ax.tick_params(axis='x', rotation=45)
        ax.tick_params(axis='y', labelsize=9)

    plt.suptitle(f"{model_label} - Per-Class Accuracy Heatmaps", fontsize=15)
    plt.tight_layout(rect=[0, 0, 1, 0.96])
    combined_path = os.path.join(OUTPUT_DIR, f"{model_key}_combined_accuracy_heatmaps.svg")
    plt.savefig(combined_path, format="svg")
    plt.close()
    print(f"Saved combined: {combined_path}")

    # INDIVIDUAL PLOTS
    for mode_key, mode_label in TRAINING_MODES.items():
        heatmap_data = np.full((len(class_names), len(DATASET_ORDER)), np.nan)

        for row_idx, cls in enumerate(class_names):
            for col_idx, ds in enumerate(DATASET_ORDER):
                try:
                    acc_list = data[cls][model_key][mode_key][ds]["accuracy"]
                    heatmap_data[row_idx][col_idx] = np.mean(acc_list)
                except KeyError:
                    continue

        plt.figure(figsize=(10, len(class_names) * 0.5 + 2))
        sns.heatmap(
            heatmap_data,
            annot=True,
            fmt=".2f",
            xticklabels=DATASET_ORDER,
            yticklabels=class_names,
            cmap="YlGnBu",
            linewidths=0.5,
            linecolor='gray',
            annot_kws={"fontsize": 8}
        )
        plt.title(f"{model_label} - {mode_label} Accuracy Heatmap", fontsize=14)
        plt.xticks(rotation=45)
        plt.yticks(fontsize=9)
        plt.tight_layout()

        indiv_path = os.path.join(OUTPUT_DIR, f"{model_key}_{mode_key}_accuracy_heatmap.svg")
        plt.savefig(indiv_path, format="svg")
        plt.close()
        print(f"Saved individual: {indiv_path}")