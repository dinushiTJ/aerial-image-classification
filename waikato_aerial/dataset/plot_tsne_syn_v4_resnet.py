import numpy as np
import os
from sklearn.manifold import TSNE
from datasets import load_dataset
from PIL import Image
import torch
import random
import torchvision.transforms as transforms
import torch.nn as nn
from torchvision import models
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
BASE_OUTPUT_DIR = "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v4_unet"
COMBINED_OUTPUT_PATH = os.path.join(BASE_OUTPUT_DIR, "tsne_combined.svg")
os.makedirs(BASE_OUTPUT_DIR, exist_ok=True)

# --- RESNET34-UNET MODEL ---
class ResNet34UNetEncoder(nn.Module):
    def __init__(self):
        super().__init__()
        base_model = models.resnet34(weights=None)
        self.encoder = nn.Sequential(*list(base_model.children())[:-2])

    def forward(self, x):
        return self.encoder(x)

# --- Load model weights ---
model = ResNet34UNetEncoder().cuda()
state_dict = torch.load("/home/dj191/Downloads/FLAIR-INC_rgbie_15cl_resnet34-unet_weights.pth", map_location="cuda")
model.load_state_dict(state_dict, strict=False)
model.eval()

# --- Image preprocessing ---
def to_3channel(img):
    if img.mode != 'RGB':
        img = img.convert('RGB')
    return img

preprocess = transforms.Compose([
    transforms.Lambda(to_3channel),
    transforms.Resize((256, 256)),
    transforms.ToTensor(),
    transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
])

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
            img = img.convert("RGB")
            img_tensor = preprocess(img).unsqueeze(0).cuda()
            with torch.no_grad():
                feat_map = model(img_tensor)
                feat = feat_map.mean(dim=(2, 3)).squeeze().cpu().numpy()
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