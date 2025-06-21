import numpy as np
import os
from openTSNE import TSNE  # pip install opentsne
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
BASE_OUTPUT_DIR = "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v5_fittransform"
COMBINED_OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, "tsne_combined.svg")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- CLIP MODEL SETUP ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").eval().cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

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
    return np.array(features), np.array(labels)

# --- Colors for classes ---
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a"
]

# --- Extract features for all datasets ---
features_by_domain = {}
labels_by_domain = {}

for ds_label, ds_name in DATASET_ORDER:
    feats, labs = extract_features(ds_name)
    features_by_domain[ds_label] = feats
    labels_by_domain[ds_label] = labs

# --- t-SNE fit on REAL dataset only ---
tsne = TSNE(n_components=2, perplexity=30, initialization="pca", random_state=42)
real_embeddings = tsne.fit(features_by_domain["real"])

# --- Transform other datasets to same space ---
projected_embeddings = {"real": real_embeddings}
for ds_label, feats in features_by_domain.items():
    if ds_label == "real":
        continue
    projected_embeddings[ds_label] = tsne.transform(feats)

# --- Plot individual dataset t-SNE ---
for ds_label, embeddings in projected_embeddings.items():
    plt.figure(figsize=(7, 5))
    labels = labels_by_domain[ds_label]
    for cls in range(NUM_CLASSES):
        idxs = labels == cls
        plt.scatter(embeddings[idxs, 0], embeddings[idxs, 1], color=colors[cls], label=f"Class {cls}", alpha=0.7, s=30)
    plt.title(f"t-SNE: {ds_label}")
    plt.legend(fontsize=8)
    plt.tight_layout()
    indiv_path = os.path.join(BASE_OUTPUT_DIR, f"tsne_{ds_label}.svg")
    plt.savefig(indiv_path, format="svg")
    plt.close()
    print(f"Saved to {indiv_path}")

# --- Combined plot ---
num_plots = len(DATASET_ORDER)
cols = 3
rows = (num_plots + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
axs = axs.flatten()

for idx, (ds_label, _) in enumerate(DATASET_ORDER):
    embeddings = projected_embeddings[ds_label]
    labels = labels_by_domain[ds_label]
    ax = axs[idx]
    for cls in range(NUM_CLASSES):
        idxs = labels == cls
        ax.scatter(embeddings[idxs, 0], embeddings[idxs, 1], color=colors[cls], label=f"Class {cls}", alpha=0.7, s=20)
    ax.set_title(ds_label, fontsize=14)
    ax.legend(fontsize=6, markerscale=0.5, loc="upper right")

# Remove unused axes
for j in range(num_plots, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
fig.savefig(COMBINED_OUTPUT_PATH, format="svg")
print(f"Combined figure saved to {COMBINED_OUTPUT_PATH}")
