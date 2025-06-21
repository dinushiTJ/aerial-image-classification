import torch
from torchvision.models import inception_v3
from torchvision import transforms
from datasets import load_dataset
from scipy.stats import shapiro
import pingouin as pg
import numpy as np
from torch.nn.functional import adaptive_avg_pool2d
from tqdm import tqdm

# Load and prepare Inception v3
inception = inception_v3(pretrained=True, transform_input=False, aux_logits=False)
inception.fc = torch.nn.Identity()  # remove classification head
inception.eval()

# Define transform
transform = transforms.Compose([
    transforms.Resize((299, 299)),
    transforms.ToTensor(),
    transforms.Normalize([0.5]*3, [0.5]*3)
])

# Load dataset from Hugging Face (replace with your dataset path and image column)
dataset = load_dataset("your/dataset", split="train")
image_column = "image"  # change if necessary

# Extract features
features = []
for item in tqdm(dataset):
    image = transform(item[image_column].convert("RGB")).unsqueeze(0)
    with torch.no_grad():
        feat = inception(image)
        features.append(feat.squeeze().numpy())

features = np.stack(features)

# Mardia's multivariate normality test (via Pingouin)
print("\nRunning Mardia's Test:")
mardia = pg.multivariate_normality(features, alpha=0.05)
print(mardia)

# Shapiro-Wilk on first few dimensions (warning: doesn't scale to high dims)
print("\nRunning Shapiro-Wilk on first 5 feature dims:")
for i in range(5):
    stat, p = shapiro(features[:, i])
    print(f"Dim {i}: W={stat:.4f}, p={p:.4e}")
