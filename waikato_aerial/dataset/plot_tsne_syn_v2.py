from datasets import load_dataset
from sklearn.manifold import TSNE
import torch
import numpy as np
import os
import random
from PIL import Image
from transformers import CLIPProcessor, CLIPModel

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
BASE_OUTPUT_DIR = "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn"

# --- CLIP MODEL SETUP ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- Load Real Dataset First ---
real_features = []
real_labels = []

print("Loading: real")
ds_real = load_dataset("dushj98/aerial_real_only", split=SPLIT)
class_buckets = {i: [] for i in range(NUM_CLASSES)}
for item in ds_real:
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
        real_features.append(feat)
        real_labels.append(cls)

# --- Compare Each Dataset to Real Individually ---
real_features = np.array(real_features)
real_labels = np.array(real_labels)

for ds_label, ds_name in DATASET_ORDER[1:]:
    syn_features = []
    syn_labels = []

    print(f"Loading: {ds_label}")
    ds = load_dataset(ds_name, split=SPLIT)

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
            syn_features.append(feat)
            syn_labels.append(cls)

    # Combine with real
    features = np.concatenate([real_features, syn_features], axis=0)
    domain_labels = np.array(["real"] * len(real_features) + [ds_label] * len(syn_features))

    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    embeddings = tsne.fit_transform(features)

    # --- PLOT ---
    plt.figure(figsize=(10, 7))
    for domain in np.unique(domain_labels):
        indices = domain_labels == domain
        plt.scatter(embeddings[indices, 0], embeddings[indices, 1], label=domain, alpha=0.6, s=30)

    plt.legend(title="Dataset")
    plt.title(f"t-SNE: real vs {ds_label}")
    output_path = os.path.join(BASE_OUTPUT_DIR, f"tsne_real_vs_{ds_label}.svg")
    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    plt.tight_layout()
    plt.savefig(output_path, format="svg")
    plt.close()
    print(f"Saved to {output_path}")