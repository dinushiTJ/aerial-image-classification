import numpy as np
import os
from sklearn.manifold import TSNE
from datasets import load_dataset
from PIL import Image
import torch
import random
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from prithvi_mae import pretrain_mae_vit_base_patch16_dec512d8b  # from repo

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
BASE_OUTPUT_DIR = "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v4_prith_v2"
COMBINED_OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, "tsne_combined.svg")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- Load Prithvi 2.0 MAE model manually ---
model = pretrain_mae_vit_base_patch16_dec512d8b()
ckpt = torch.load("/path/to/Prithvi_EO_V2_600M.pt", map_location="cpu")
model.load_state_dict(ckpt["model"])
model.eval().cuda()

# --- Image Preprocessing ---
import torchvision.transforms as T
transform = T.Compose([
    T.Resize((224, 224)),
    T.ToTensor(),
    T.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
])

# --- Feature Extraction ---
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
            img = transform(img).unsqueeze(0).to("cuda")
            with torch.no_grad():
                feat = model.forward_encoder(img)[0].mean(dim=1).cpu().numpy().squeeze()
            features.append(feat)
            labels.append(cls)
    return np.array(features), labels

# --- Distinct Colors ---
colors = [
    "#1f77b4", "#ff7f0e", "#2ca02c", "#d62728", "#9467bd", "#8c564b",
    "#e377c2", "#7f7f7f", "#bcbd22", "#17becf", "#aec7e8", "#ffbb78", "#98df8a"
]

# --- Plotting Loop ---
num_plots = len(DATASET_ORDER)
cols = 3
rows = (num_plots + cols - 1) // cols
fig, axs = plt.subplots(rows, cols, figsize=(cols * 5, rows * 5))
axs = axs.flatten()

for idx, (ds_label, ds_name) in enumerate(DATASET_ORDER):
    features, labels = extract_features(ds_name)

    tsne = TSNE(n_components=2, perplexity=30, init='pca', random_state=42)
    embeddings = tsne.fit_transform(features)

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

    ax = axs[idx]
    for cls in range(NUM_CLASSES):
        indices = np.array(labels) == cls
        ax.scatter(embeddings[indices, 0], embeddings[indices, 1], color=colors[cls], label=f"Class {cls}", alpha=0.7, s=20)
    ax.set_title(f"{ds_label}", fontsize=14)
    ax.legend(fontsize=8, loc='upper right', markerscale=0.5)

for j in range(num_plots, len(axs)):
    fig.delaxes(axs[j])

plt.tight_layout()
fig.savefig(COMBINED_OUTPUT_PATH, format="svg")
print(f"Combined figure saved to {COMBINED_OUTPUT_PATH}")