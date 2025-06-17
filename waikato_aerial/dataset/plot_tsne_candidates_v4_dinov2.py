from datasets import load_dataset
from sklearn.manifold import TSNE
import torch
import numpy as np
import os
import random
from PIL import Image
from transformers import AutoImageProcessor, AutoModel

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt

# --- REPRODUCIBILITY ---
torch.manual_seed(42)
np.random.seed(42)
random.seed(42)

# --- CONFIG ---
DATASET_ORDER = [
    ("real", "dushj98/waikato_aerial_imagery_2017"),
    ("v0", "dushj98/waikato_aerial_2017_synthetic_v0"),
    ("v1", "dushj98/waikato_aerial_2017_synthetic_v1"),
    ("v2", "dushj98/waikato_aerial_2017_synthetic_v2"),
    ("ti_v1", "dushj98/waikato_aerial_2017_synthetic_ti_v1"),
    ("ti_v2", "dushj98/waikato_aerial_2017_synthetic_ti_v2"),
    ("v0_up", "dushj98/waikato_aerial_2017_synthetic_v0_upscaled"),
    ("v1_up", "dushj98/waikato_aerial_2017_synthetic_v1_upscaled"),
    ("v2_up", "dushj98/waikato_aerial_2017_synthetic_v2_upscaled"),
    ("ti_v1_up", "dushj98/waikato_aerial_2017_synthetic_ti_v1_upscaled"),
    ("ti_v2_up", "dushj98/waikato_aerial_2017_synthetic_ti_v2_upscaled"),
    ("best_cmmd", "dushj98/waikato_aerial_2017_synthetic_best_cmmd"),
    ("best_fid", "dushj98/waikato_aerial_2017_synthetic_best_fid")
]
NUM_CLASSES = 13
SAMPLES_PER_CLASS = 10
SPLIT = "train"
BASE_OUTPUT_DIR = "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v4_dinov2"
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- CLIP MODEL SETUP ---
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-base")
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-base").eval().cuda()

# --- Color Setup: same 13 distinct colors ---
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

# --- FEATURE EXTRACTION ---
def extract_features(dataset_name):
    print(f"Loading: {dataset_name}")
    ds = load_dataset(dataset_name, split=SPLIT)
    features, labels = [], []
    buckets = {i: [] for i in range(NUM_CLASSES)}

    for item in ds:
        if len(buckets[item['label']]) < SAMPLES_PER_CLASS:
            buckets[item['label']].append(item)
        if all(len(v) >= SAMPLES_PER_CLASS for v in buckets.values()):
            break

    for cls, samples in buckets.items():
        for sample in samples:
            img = sample["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            inputs = dinov2_processor(images=img, return_tensors="pt").to("cuda")
            with torch.no_grad():
                feat = dinov2_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
            features.append(feat)
            labels.append(cls)

    return np.array(features), np.array(labels)

# --- PLOT FUNCTION ---
def plot_tsne(embeddings, labels, title, path, figsize=(6,5), scatter_size=20, alpha=0.7, legend=True, ax=None):
    if ax is None:
        plt.figure(figsize=figsize)
        ax = plt.gca()
        created_fig = True
    else:
        created_fig = False

    for cls in range(NUM_CLASSES):
        idx = labels == cls
        ax.scatter(embeddings[idx, 0], embeddings[idx, 1], 
                   color=colors[cls], label=f"Class {cls}", s=scatter_size, alpha=alpha)
    ax.set_title(title, fontsize=14 if not created_fig else 16)
    ax.set_xticks([])
    ax.set_yticks([])
    if legend:
        ax.legend(fontsize=10 if not created_fig else 12, loc='best')
    if created_fig:
        plt.tight_layout()
        plt.savefig(path, format="svg")
        plt.close()

# --- MAIN LOOP ---
n = len(DATASET_ORDER)
cols = 3
rows = (n + cols - 1) // cols
fig, axes = plt.subplots(rows, cols, figsize=(cols * 6, rows * 5))
axes = axes.flatten()

for idx, (label, ds_name) in enumerate(DATASET_ORDER):
    feats, labs = extract_features(ds_name)

    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    emb = tsne.fit_transform(feats)

    # Save individual plot immediately
    indiv_path = os.path.join(BASE_OUTPUT_DIR, f"tsne_{label}_solo.svg")
    plot_tsne(emb, labs, f"t-SNE: {label}", indiv_path, legend=True)

    # Add to combined plot without legend (to avoid clutter)
    ax = axes[idx]
    plot_tsne(emb, labs, f"{label}", None, ax=ax, legend=False)

# Remove unused axes
for j in range(idx + 1, len(axes)):
    fig.delaxes(axes[j])

plt.tight_layout()
combined_path = os.path.join(BASE_OUTPUT_DIR, "all_tsne_individuals.svg")
fig.savefig(combined_path, format="svg")
plt.close()

print(f"Saved individual plots and combined grid to {BASE_OUTPUT_DIR}")
