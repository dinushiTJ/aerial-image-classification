import numpy as np
import os
from openTSNE import TSNE
from datasets import load_dataset
from PIL import Image
from transformers import CLIPProcessor, CLIPModel
import torch
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# Reproducibility
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# Config
REAL_DATASET = ("real", "dushj98/aerial_real_only")
SYNTHETIC_DATASETS = [
    ("p10",   "dushj98/aerial_real_plus_0010"),
    ("p25",   "dushj98/aerial_real_plus_0025"),
    ("p50",   "dushj98/aerial_real_plus_0050"),
    ("p75",   "dushj98/aerial_real_plus_0075"),
    ("p100",  "dushj98/aerial_real_plus_0100"),
    ("p125",  "dushj98/aerial_real_plus_0125"),
    ("p150",  "dushj98/aerial_real_plus_0150"),
]
NUM_CLASSES = 13
SAMPLES_PER_CLASS = 2
SPLIT = "train"
OUTPUT_DIR = f"/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v5_fittransform_n{SAMPLES_PER_CLASS}_v3"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# CLIP Model Setup
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").eval().cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

# Feature extraction
def extract_features(dataset_name):
    features, labels = [], []
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

# Colors
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a"
]

# Extract and embed real dataset
real_feats, real_labels = extract_features(REAL_DATASET[1])
tsne = TSNE(n_components=2, perplexity=30, initialization="pca", random_state=42, verbose=True)
embedding_real = tsne.fit(real_feats)

# Plot and save real-only embedding
fig_real, ax_real = plt.subplots(figsize=(10, 8))
for cls in range(NUM_CLASSES):
    idxs = real_labels == cls
    ax_real.scatter(
        embedding_real[idxs, 0], embedding_real[idxs, 1],
        color=colors[cls], label=f"Class {cls}", alpha=0.8, s=30, marker='o'
    )
ax_real.set_title("t-SNE: Real Dataset")
ax_real.legend(fontsize=9, markerscale=1.0, loc='best', ncol=2)
fig_real.tight_layout()
real_path = os.path.join(OUTPUT_DIR, "tsne_real_only.svg")
fig_real.savefig(real_path, format="svg")
plt.close(fig_real)

# Store plot paths
plot_paths = [real_path]

# Generate synthetic plots
for syn_label, syn_name in SYNTHETIC_DATASETS:
    syn_feats, syn_labels = extract_features(syn_name)
    embedding_syn = embedding_real.transform(syn_feats)

    fig, ax = plt.subplots(figsize=(10, 8))
    for cls in range(NUM_CLASSES):
        idxs = syn_labels == cls
        ax.scatter(
            embedding_syn[idxs, 0], embedding_syn[idxs, 1],
            color=colors[cls], label=f"Class {cls}", alpha=0.8, s=30, marker='x'
        )
    ax.set_title(f"t-SNE: Augmented Dataset ({syn_label})")
    ax.legend(fontsize=9, markerscale=1.0, loc='best', ncol=2)
    fig.tight_layout()
    syn_path = os.path.join(OUTPUT_DIR, f"tsne_aug_{syn_label}.svg")
    fig.savefig(syn_path, format="svg")
    plt.close(fig)
    plot_paths.append(syn_path)

# Create composite figure
fig, axes = plt.subplots(nrows=int(np.ceil(len(plot_paths) / 3)), ncols=3, figsize=(20, 14))
axes = axes.flatten()

for i, path in enumerate(plot_paths):
    img = Image.open(path)
    axes[i].imshow(img)
    axes[i].axis("off")

for j in range(len(plot_paths), len(axes)):
    axes[j].axis("off")

# Save final composite
plt.tight_layout()
composite_path = os.path.join(OUTPUT_DIR, "tsne_composite.svg")
plt.savefig(composite_path, format="svg", bbox_inches="tight")
plt.close()
print(f"Saved composite plot: {composite_path}")
