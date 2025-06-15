from huggingface_hub import HfApi
from datasets import get_dataset_config_names
import matplotlib.pyplot as plt
import os

# Define datasets in desired order and HF repo IDs
DATASET_ORDER = ["real", "p10", "p25", "p50", "p75", "p100", "p125", "p150"]
HF_IDS = {
    "real":  "dushj98/aerial_real_only",
    "p10":   "dushj98/aerial_real_plus_0010",
    "p25":   "dushj98/aerial_real_plus_0025",
    "p50":   "dushj98/aerial_real_plus_0050",
    "p75":   "dushj98/aerial_real_plus_0075",
    "p100":  "dushj98/aerial_real_plus_0100",
    "p125":  "dushj98/aerial_real_plus_0125",
    "p150":  "dushj98/aerial_real_plus_0150",
}

api = HfApi()
split = "train"
output_path = "/home/dj191/research/code/waikato_aerial/dataset/plots/dataset_size_bar.svg"
dataset_counts = []

print("Starting dataset size extraction...\n")

for label in DATASET_ORDER:
    repo_id = HF_IDS[label]
    print(f"Processing dataset: {label} ({repo_id})")

    try:
        print(f"  Fetching dataset info...")
        info = api.dataset_info(repo_id, files_metadata=False)

        print(f"  Loading '{split}' split with streaming...")
        from datasets import load_dataset
        ds = load_dataset(repo_id, split=split, streaming=True)

        print(f"  Counting examples...")
        count = sum(1 for _ in ds)
        print(f"  -> Found {count} examples.")
    except Exception as e:
        print(f"  !!! Error fetching train size for {repo_id}: {e}")
        count = 0

    dataset_counts.append((label, count))

print("\nAll dataset sizes extracted.")
print("Creating bar chart...")

# Plotting
labels, sizes = zip(*dataset_counts)
plt.figure(figsize=(10, 6))
bars = plt.bar(labels, sizes, color="steelblue")
plt.xticks(rotation=45, ha="right")
plt.ylabel("Number of Training Examples")
plt.xlabel("Datasets")
plt.title("Training Split Sizes Across Datasets")

for bar, val in zip(bars, sizes):
    plt.text(bar.get_x() + bar.get_width()/2, val + max(sizes)*0.01, str(val),
             ha='center', va='bottom', fontsize=8)

os.makedirs(os.path.dirname(output_path), exist_ok=True)
plt.tight_layout()
plt.savefig(output_path, format="svg")
plt.close()

print(f"Plot saved to {output_path}")
