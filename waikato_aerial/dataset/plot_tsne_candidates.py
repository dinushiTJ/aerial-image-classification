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
OUTPUT_PATH = "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_combined.svg"

# --- CLIP MODEL SETUP ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").eval().cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# --- FEATURE COLLECTION ---
features, labels, domains = [], [], []

for domain_label, (ds_label, ds_name) in enumerate(DATASET_ORDER):
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
            features.append(feat)
            labels.append(cls)
            domains.append(ds_label)

# --- TSNE ---
x = np.array(features)
y = np.array(labels)
d = np.array(domains)
tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
embeddings = tsne.fit_transform(x)

# --- PLOT ---
plt.figure(figsize=(12, 8))
colors = plt.cm.get_cmap("tab10", NUM_CLASSES)

for domain in np.unique(d):
    indices = d == domain
    plt.scatter(embeddings[indices, 0], embeddings[indices, 1], label=domain, alpha=0.6, s=30)

plt.legend(title="Dataset")
plt.title("t-SNE of CLIP Image Features from Different Datasets")
os.makedirs(os.path.dirname(OUTPUT_PATH), exist_ok=True)
plt.tight_layout()
plt.savefig(OUTPUT_PATH, format="svg")
plt.close()
print(f"Saved t-SNE plot to {OUTPUT_PATH}")