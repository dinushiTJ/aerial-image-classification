from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import random

# Configs
dataset_names = [
    "dushj98/waikato_aerial_imagery_2017",
    "dushj98/waikato_aerial_2017_synthetic_v0",
    "dushj98/waikato_aerial_2017_synthetic_v1",
    "dushj98/waikato_aerial_2017_synthetic_v2",
    "dushj98/waikato_aerial_2017_synthetic_ti_v1",
    "dushj98/waikato_aerial_2017_synthetic_ti_v2",
    "dushj98/waikato_aerial_2017_synthetic_v0_upscaled",
    "dushj98/waikato_aerial_2017_synthetic_v1_upscaled",
    "dushj98/waikato_aerial_2017_synthetic_v2_upscaled",
    "dushj98/waikato_aerial_2017_synthetic_ti_v1_upscaled",
    "dushj98/waikato_aerial_2017_synthetic_ti_v2_upscaled",
    "dushj98/waikato_aerial_2017_synthetic_best_cmmd",
    "dushj98/waikato_aerial_2017_synthetic_best_fid"
]
num_classes = 13
output_svg_path = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/dataset/plots/dataset_grid.svg"
split = "train"

# Plot config
fig, axes = plt.subplots(nrows=len(dataset_names), ncols=num_classes, figsize=(num_classes*1.3, len(dataset_names)*1.3))
plt.subplots_adjust(wspace=0.05, hspace=0.05)

for row_idx, dataset_name in enumerate(dataset_names):
    ds = load_dataset(dataset_name, split=split)

    # For each class, pick one random image
    selected_images = []
    for class_idx in range(num_classes):
        class_imgs = [ex["image"] for ex in ds if ex["label"] == class_idx]
        if class_imgs:
            selected_images.append(random.choice(class_imgs))
        else:
            selected_images.append(None)

    for col_idx, img in enumerate(selected_images):
        ax = axes[row_idx, col_idx]
        if img:
            ax.imshow(img)
        ax.axis("off")
        if row_idx == 0:
            ax.set_title(f"Class {col_idx}", fontsize=8)
        if col_idx == 0:
            ax.set_ylabel(dataset_name.split("/")[-1], fontsize=8)

plt.tight_layout()
os.makedirs(os.path.dirname(output_svg_path), exist_ok=True)
plt.savefig(output_svg_path, format="svg")
plt.close()
