import numpy as np
import os
from sklearn.manifold import TSNE
from datasets import load_dataset
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import random

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- REPRODUCIBILITY ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- CONFIG ---
DATASET_ORDER = [
    ("real",  "dushj98/aerial_real_only"),
    ("p10",   "dushj98/aerial_real_plus_0010"),
    ("p25",   "dushj98/aerial_real_plus_0025"),
    ("p50",   "dushj98/aerial_real_plus_0050"),
    ("p75",   "dushj98/aerial_real_plus_0075"),
    ("p100",  "dushj98/aerial_real_plus_0100"),
    ("p125",  "dushj98/aerial_real_plus_0125"),
    ("p150",  "dushj98/aerial_real_plus_0150"),
]
NUM_CLASSES = 13
SAMPLES_PER_CLASS = 10
SPLIT = "train"
BASE_OUTPUT_DIR = "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v4_2"
COMBINED_OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, "tsne_combined.svg")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- CLIP MODEL SETUP ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Feature Extraction Helper ---
def extract_features(dataset_name):
    features, labels = [], []
    print(f"Loading: {dataset_name}")
    ds = load_dataset(dataset_name, split=SPLIT)
    class_buckets = {i: [] for i in range(NUM_CLASSES)}
    for item in ds:
        if len(class_buckets[item['label']]) < SAMPLES_PER_CLASS:
            class_buckets[item['label']].append(item)
        if all(len(v) >= SAMPLES_PER_CLASS for v in class_buckets.values()):
            break

    for cls, samples in class_buckets.items():
        for sample in samples:
            img = sample["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            inputs = clip_processor(images=img, return_tensors="pt").to("cuda")
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs).cpu().numpy().squeeze()
            features.append(feat)
            labels.append(cls)
    return np.array(features), labels

# --- Clear, distinct colors for 13 classes ---
colors = [
    "#1f77b4",  # muted blue
    "#ff7f0e",  # safety orange
    "#2ca02c",  # cooked asparagus green
    "#d62728",  # brick red
    "#9467bd",  # muted purple
    "#8c564b",  # chestnut brown
    "#e377c2",  # raspberry yogurt pink
    "#7f7f7f",  # middle gray
    "#bcbd22",  # curry yellow-green
    "#17becf",  # blue-teal
    "#aec7e8",  # light blue
    "#ffbb78",  # light orange
    "#98df8a",  # light green
]

# --- Plotting and main loop ---
num_plots = len(DATASET_ORDER)
cols = 3
rows = (num_plots + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
axs = axs.flatten()

for idx, (ds_label, ds_name) in enumerate(DATASET_ORDER):
    features, labels = extract_features(ds_name)

    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    embeddings = tsne.fit_transform(features)

    # Individual plot
    indiv_path = os.path.join(BASE_OUTPUT_DIR, f"tsne_{ds_label}.svg")
    plt.figure(figsize=(7, 5))
    for cls in range(NUM_CLASSES):
        indices = np.array(labels) == cls
        plt.scatter(embeddings[indices, 0], embeddings[indices, 1], color=colors[cls], label=f"Class {cls}", alpha=0.7, s=30)
    plt.title(f"t-SNE: {ds_label}", fontsize=14)
    plt.legend(fontsize=8)
    plt.tight_layout()
    plt.savefig(indiv_path, format="svg")
    plt.close()
    print(f"Saved to {indiv_path}")

    # Add to combined grid
    ax = axs[idx]
    for cls in range(NUM_CLASSES):
        indices = np.array(labels) == cls
        ax.scatter(embeddings[indices, 0], embeddings[indices, 1], color=colors[cls], label=f"Class {cls}", alpha=0.7, s=20)
    ax.set_title(f"{ds_label}", fontsize=14)
    ax.legend(fontsize=8, loc='upper right', markerscale=0.5)

# Clean unused axes
for j in range(num_plots, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
fig.savefig(COMBINED_OUTPUT_PATH, format="svg")
print(f"Combined figure saved to {COMBINED_OUTPUT_PATH}")
