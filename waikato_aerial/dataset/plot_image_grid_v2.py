from datasets import load_dataset
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import os
from concurrent.futures import ThreadPoolExecutor, as_completed

# Dataset shorthand mapping
dataset_name_map = {
    "dushj98/waikato_aerial_imagery_2017": "real",
    "dushj98/waikato_aerial_2017_synthetic_v0": "v0",
    "dushj98/waikato_aerial_2017_synthetic_v1": "v1",
    "dushj98/waikato_aerial_2017_synthetic_v2": "v2",
    "dushj98/waikato_aerial_2017_synthetic_ti_v1": "ti_v1",
    "dushj98/waikato_aerial_2017_synthetic_ti_v2": "ti_v2",
    "dushj98/waikato_aerial_2017_synthetic_v0_upscaled": "v0_up",
    "dushj98/waikato_aerial_2017_synthetic_v1_upscaled": "v1_up",
    "dushj98/waikato_aerial_2017_synthetic_v2_upscaled": "v2_up",
    "dushj98/waikato_aerial_2017_synthetic_ti_v1_upscaled": "ti_v1_up",
    "dushj98/waikato_aerial_2017_synthetic_ti_v2_upscaled": "ti_v2_up",
    "dushj98/waikato_aerial_2017_synthetic_best_cmmd": "best_cmmd",
    "dushj98/waikato_aerial_2017_synthetic_best_fid": "best_fid"
}

dataset_names = list(dataset_name_map.keys())
num_classes = 13
output_svg_path = "/home/dj191/research/code/waikato_aerial/dataset/plots/dataset_grid_v2.svg"
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

# Pull class labels from the real dataset
label_names = None
try:
    real_info = load_dataset("dushj98/waikato_aerial_imagery_2017", split=split)
    label_names = real_info.features["label"].names
except Exception as e:
    print("Failed to fetch class names, defaulting to index")
    label_names = [f"Class {i}" for i in range(num_classes)]

# Plot config
print("Plotting...")
fig = plt.figure(figsize=((num_classes+1)*1.5, len(dataset_names)*1.5))
gs = gridspec.GridSpec(len(dataset_names), num_classes+1, wspace=0.05, hspace=0.05)

for row_idx, dataset_name in enumerate(dataset_names):
    # First column with dataset name
    ax_name = fig.add_subplot(gs[row_idx, 0])
    ax_name.text(0.5, 0.5, dataset_name_map[dataset_name], fontsize=10, va='center', ha='center')
    ax_name.axis("off")

    class_images = results.get(dataset_name, {})
    for col_idx in range(num_classes):
        ax = fig.add_subplot(gs[row_idx, col_idx+1])
        img = class_images.get(col_idx)
        if img:
            ax.imshow(img)
        ax.axis("off")

        if row_idx == 0:
            ax.set_title(label_names[col_idx], fontsize=10, rotation=90, va="bottom", ha="center")

plt.tight_layout()
os.makedirs(os.path.dirname(output_svg_path), exist_ok=True)
plt.savefig(output_svg_path, format="svg")
plt.close()
print(f"Plot saved successfully to: {output_svg_path}")
