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
BASE_OUTPUT_DIR = "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v3"

# --- CLIP MODEL SETUP ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- LOAD REAL DATA FIRST ---
print("Loading: real")
real_ds = load_dataset(DATASET_ORDER[0][1], split=SPLIT)
real_features, real_labels = [], []
real_buckets = {i: [] for i in range(NUM_CLASSES)}

for item in real_ds:
    if len(real_buckets[item['label']]) < SAMPLES_PER_CLASS:
        real_buckets[item['label']].append(item)
    if all(len(v) >= SAMPLES_PER_CLASS for v in real_buckets.values()):
        break

for cls, samples in real_buckets.items():
    for sample in samples:
        img = sample["image"]
        if not isinstance(img, Image.Image):
            img = Image.fromarray(img)
        inputs = clip_processor(images=img, return_tensors="pt").to("cuda")
        with torch.no_grad():
            feat = clip_model.get_image_features(**inputs).cpu().numpy().squeeze()
        real_features.append(feat)
        real_labels.append(cls)

real_features = np.array(real_features)
real_labels = np.array(real_labels)

# --- PROCESS EACH SYNTHETIC DATASET ---
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

fig, axes = plt.subplots(nrows=(len(DATASET_ORDER) - 1) // 3 + 1, ncols=3, figsize=(18, 6*((len(DATASET_ORDER)-1)//3 + 1)))
axes = axes.flatten()

for idx, (ds_label, ds_name) in enumerate(DATASET_ORDER[1:]):
    print(f"Comparing: real vs {ds_label}")
    syn_features, syn_labels = [], []
    ds = load_dataset(ds_name, split=SPLIT)
    syn_buckets = {i: [] for i in range(NUM_CLASSES)}

    for item in ds:
        if len(syn_buckets[item['label']]) < SAMPLES_PER_CLASS:
            syn_buckets[item['label']].append(item)
        if all(len(v) >= SAMPLES_PER_CLASS for v in syn_buckets.values()):
            break

    for cls, samples in syn_buckets.items():
        for sample in samples:
            img = sample["image"]
            if not isinstance(img, Image.Image):
                img = Image.fromarray(img)
            inputs = clip_processor(images=img, return_tensors="pt").to("cuda")
            with torch.no_grad():
                feat = clip_model.get_image_features(**inputs).cpu().numpy().squeeze()
            syn_features.append(feat)
            syn_labels.append(cls)

    x = np.concatenate([real_features, np.array(syn_features)], axis=0)
    domains = np.array(["real"] * len(real_features) + [ds_label] * len(syn_features))

    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    embeddings = tsne.fit_transform(x)

    ax = axes[idx]
    for domain in np.unique(domains):
        idxs = domains == domain
        ax.scatter(embeddings[idxs, 0], embeddings[idxs, 1], label=domain, alpha=0.7, s=20)
    ax.set_title(f"Real vs {ds_label}")
    ax.set_xticks([])
    ax.set_yticks([])

    plt.figure()
    plt.scatter(embeddings[domains == 'real', 0], embeddings[domains == 'real', 1], label='real', alpha=0.7, s=30)
    plt.scatter(embeddings[domains == ds_label, 0], embeddings[domains == ds_label, 1], label=ds_label, alpha=0.7, s=30)
    plt.legend()
    plt.title(f"t-SNE: Real vs {ds_label}")
    plt.tight_layout()
    save_path = os.path.join(BASE_OUTPUT_DIR, f"tsne_{ds_label}.svg")
    plt.savefig(save_path, format="svg")
    plt.close()
    print(f"Saved: {save_path}")

# Final merged figure save
plt.tight_layout()
plt.savefig(os.path.join(BASE_OUTPUT_DIR, "all_tsne.svg"), format="svg")
plt.close()
