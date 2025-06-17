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
OUTPUT_PATH = "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v1/tsne_training_datasets_combined.svg"

# --- CLIP MODEL SETUP ---
clip_model = CLIPModel.from_pretrained("openai/clip-vit-large-patch14-336").eval().cuda()
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-large-patch14-336")

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