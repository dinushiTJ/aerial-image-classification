import numpy as np
import os
from openTSNE import TSNE
from datasets import load_dataset
from PIL import Image
from transformers import AutoImageProcessor, AutoModel
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
REAL_DATASET = ("real", "dushj98/aerial_real_only")
SYNTHETIC_DATASETS = [
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
SAMPLES_PER_CLASS = 666
SPLIT = "train"
OUTPUT_DIR = f"/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v5_fittransform_n{SAMPLES_PER_CLASS}_v2_dino"
os.makedirs(OUTPUT_DIR, exist_ok=True)

# --- Colors for classes ---
COLORS = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a"
]

# --- CLIP MODEL SETUP ---
dinov2_processor = AutoImageProcessor.from_pretrained("facebook/dinov2-giant", use_fast=True)
dinov2_model = AutoModel.from_pretrained("facebook/dinov2-giant").eval().cuda()

# --- Feature Extraction Helper ---
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
            inputs = dinov2_processor(images=img, return_tensors="pt").to("cuda")
            with torch.no_grad():
                feat = dinov2_model(**inputs).last_hidden_state.mean(dim=1).cpu().numpy().squeeze()
            features.append(feat)
            labels.append(cls)
    return np.array(features), np.array(labels)


def plot_individual_graph(embeddings, labels, aug_name = "") -> None:
    if aug_name:
        title = f"t-SNE: Synthetic Dataset {aug_name}"
        path_suffix = aug_name.lower().replace(" ", "_").replace("-", "_")
    else:
        title = "t-SNE: Real Dataset"
        path_suffix = "real"


    fig_real, ax_real = plt.subplots(figsize=(10, 8))
    for cls in range(NUM_CLASSES):
        idxs = labels == cls
        ax_real.scatter(
            embeddings[idxs, 0], embeddings[idxs, 1],
            color=COLORS[cls], label=f"Class {cls}", alpha=0.8, s=30, marker='o'
        )
    ax_real.set_title(title)
    ax_real.legend(fontsize=9, markerscale=1.0, loc='best', ncol=2)
    fig_real.tight_layout()
    real_path = os.path.join(OUTPUT_DIR, f"tsne_{path_suffix}.svg")
    fig_real.savefig(real_path, format="svg")
    plt.close(fig_real)

# --- Extract features for real dataset once ---
real_feats, real_labels = extract_features(REAL_DATASET[1])

# Fit t-SNE on REAL data only
tsne = TSNE(
    n_components=2,
    perplexity=30,
    initialization="pca",
    random_state=42,
    verbose=True,
)
embedding_real = tsne.fit(real_feats)

# plot real only
plot_individual_graph(embedding_real, real_labels) 

# --- Loop over synthetic datasets ---
for syn_label, syn_name in SYNTHETIC_DATASETS:
    print(f"Processing pair: real vs {syn_label}")

    syn_feats, syn_labels = extract_features(syn_name)

    # Transform SYNTHETIC data to the real t-SNE embedding space
    embedding_syn = embedding_real.transform(syn_feats)
    plot_individual_graph(embedding_syn, syn_labels, aug_name=syn_label)

    # Plot combined embedding
    fig, ax = plt.subplots(figsize=(10, 8))

    # Plot real data with lighter alpha
    for cls in range(NUM_CLASSES):
        idxs = real_labels == cls
        ax.scatter(
            embedding_real[idxs, 0], embedding_real[idxs, 1],
            color=COLORS[cls], label=f"Real Class {cls}", alpha=0.3, s=30, marker='o'
        )

    # Plot synthetic data with stronger alpha
    for cls in range(NUM_CLASSES):
        idxs = syn_labels == cls
        ax.scatter(
            embedding_syn[idxs, 0], embedding_syn[idxs, 1],
            color=COLORS[cls], label=f"Syn Class {cls}", alpha=0.7, s=30, marker='x'
        )

    ax.set_title(f"t-SNE: Real vs Synthetic Datasets ({syn_label})")
    ax.legend(fontsize=9, markerscale=1.0, loc='best', ncol=2)
    fig.tight_layout()
    outpath = os.path.join(OUTPUT_DIR, f"tsne_real_vs_syn_{syn_label}.svg")
    fig.savefig(outpath, format="svg")
    plt.close(fig)
    print(f"Saved plot: {outpath}")

