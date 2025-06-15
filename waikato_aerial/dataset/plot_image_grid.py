from datasets import load_dataset
import matplotlib.pyplot as plt
import os
import random
from concurrent.futures import ThreadPoolExecutor, as_completed

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
output_svg_path = "/home/dj191/research/code/waikato_aerial/dataset/plots/dataset_grid.svg"
split = "train"

# Util: Load and fetch one image per class
def load_one_sample_per_class(dataset_name):
    try:
        print(f"Loading: {dataset_name}")
        ds = load_dataset(dataset_name, split=split, streaming=True)
        class_samples = {}

        for ex in ds:
            label = ex["label"]
            if label not in class_samples and 0 <= label < num_classes:
                class_samples[label] = ex["image"]
                if len(class_samples) == num_classes:
                    break
        print(f"  -> Collected {len(class_samples)} samples from {dataset_name}")
        return dataset_name, class_samples
    except Exception as e:
        print(f"  !!! Failed to load {dataset_name}: {e}")
        return dataset_name, {}

# Async load
print("Starting concurrent dataset processing...")
results = {}
with ThreadPoolExecutor(max_workers=6) as executor:
    future_to_name = {executor.submit(load_one_sample_per_class, name): name for name in dataset_names}
    for future in as_completed(future_to_name):
        name = future_to_name[future]
        _, class_images = future.result()
        results[name] = class_images

# Plot config
print("Plotting...")
fig, axes = plt.subplots(nrows=len(dataset_names), ncols=num_classes, figsize=(num_classes*1.3, len(dataset_names)*1.3))
plt.subplots_adjust(wspace=0.05, hspace=0.05)

for row_idx, dataset_name in enumerate(dataset_names):
    class_images = results.get(dataset_name, {})
    for col_idx in range(num_classes):
        ax = axes[row_idx, col_idx]
        img = class_images.get(col_idx)
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
print(f"Plot saved successfully to: {output_svg_path}")
