import math
import os
import pickle
import random
from collections import defaultdict

import click
from datasets import ClassLabel
from datasets import Dataset
from datasets import DatasetDict
from datasets import concatenate_datasets
from datasets import load_dataset

TARGET_PER_CLASS = 666
FRACTIONS = [0.0, 0.1, 0.25, 0.5, 0.75, 1.0, 1.25, 1.5]


def group_by_class(dataset: Dataset) -> dict[int, list[dict]]:
    class_map = defaultdict(list)
    for example in dataset:
        class_map[example["label"]].append(example)
    return class_map


def sample_class_subset(class_samples: list[dict], n: int, seed: int) -> list[dict]:
    rnd = random.Random(seed)
    samples = class_samples.copy()
    rnd.shuffle(samples)
    return samples[:n]


def create_and_push_merged_dataset(
    real_grouped: dict,
    syn_grouped: dict,
    frac: float,
    val_split: Dataset,
    repo_prefix: str,
    huggingface_token: str,
    real_label_features: ClassLabel,
    syn_label_features: ClassLabel,
    seed: int,
):
    if (
        real_label_features["label"].num_classes
        != syn_label_features["label"].num_classes
    ):
        raise ValueError(
            f"Num classes are inequal in real ({real_label_features['label'].num_classes}) and synthetic ({syn_label_features['label'].num_classes}) datasets."
        )

    grouped_real_samples: dict[list[dict]] = defaultdict(list)
    grouped_syn_samples: dict[list[dict]] = defaultdict(list)

    for label in list(real_grouped.keys()):
        # Add all real samples
        real_samples = real_grouped[label]
        if len(real_samples) != TARGET_PER_CLASS:
            raise ValueError(
                f"{real_label_features.int2str(label)} size {len(real_samples)} does not match the target: {TARGET_PER_CLASS}"
            )

        grouped_real_samples[label].extend(real_samples)

    for label in list(syn_grouped.keys()):
        # Sample and add synthetic samples
        if frac > 0 and label in syn_grouped:
            syn_samples = syn_grouped[label]
            syn_count = math.floor(TARGET_PER_CLASS * frac)
            syn_subset = sample_class_subset(
                class_samples=syn_samples, n=syn_count, seed=seed
            )
            grouped_syn_samples[label].extend(syn_subset)

    print(
        f"Sampling Summary\n---\nReal Classes ({len(grouped_real_samples.keys())}): {grouped_real_samples.keys()}"
    )
    print(
        f"Synthetic Classes ({len(grouped_syn_samples.keys())}): {grouped_syn_samples.keys()}\n\n"
    )

    # Flatten and convert to HuggingFace Datasets
    processed_real_ds = Dataset.from_list(
        [item for sublist in grouped_real_samples.values() for item in sublist]
    ).cast(real_label_features)

    if frac > 0:
        processed_syn_ds = Dataset.from_list(
            [item for sublist in grouped_syn_samples.values() for item in sublist]
        ).cast(syn_label_features)

        merged_train = concatenate_datasets([processed_real_ds, processed_syn_ds])
    else:
        merged_train = processed_real_ds

    # Shuffle to mix real and synthetic
    merged_train = merged_train.shuffle(seed=seed)

    # Create DatasetDict and push
    merged_ds_dict = DatasetDict({"train": merged_train, "validation": val_split})

    suffix = (
        "real_only" if frac == 0.0 else f"real_plus_{str(int(frac * 100)).zfill(4)}"
    )
    repo_id = repo_prefix + suffix

    print(f"Pushing {repo_id} to Hugging Face Hub...")
    merged_ds_dict.push_to_hub(repo_id, token=huggingface_token)
    print(f"✅ Done: {repo_id}\n\n")


@click.command()
@click.option(
    "--real",
    "-r",
    default="dushj98/waikato_aerial_imagery_2017",
    help="HuggingFace real dataset name (e.g., dushj98/waikato_aerial_imagery_2017)",
)
@click.option(
    "--synthetic",
    "-s",
    default="dushj98/waikato_aerial_2017_synthetic_best_cmmd",
    help="HuggingFace synthetic dataset name",
)
@click.option("--huggingface-token", "-h", required=True, help="HuggingFace token")
@click.option(
    "--repo-prefix",
    "-p",
    default="dushj98/aerial_",
    help="Prefix for pushing merged datasets to HF Hub (e.g., dushj98/aerial_)",
)
@click.option(
    "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
)
def merge(
    real: str, synthetic: str, huggingface_token: str, repo_prefix: str, seed: int
):
    print(f"Loading real dataset (train + validation): {real}")
    real = load_dataset(real, token=huggingface_token)

    print(f"Loading synthetic dataset (train only): {synthetic}")
    syn_train = load_dataset(synthetic, split="train", token=huggingface_token)

    # Load or create cache
    os.makedirs(".cache", exist_ok=True)
    real_cache_path = ".cache/real_grouped.pkl"
    syn_cache_path = ".cache/syn_grouped.pkl"

    try:
        print("🔁 Loading cached real_grouped...")
        with open(real_cache_path, "rb") as f:
            real_grouped = pickle.load(f)
    except Exception:
        print("Grouping real dataset by class...")
        real_grouped = group_by_class(real["train"])
        with open(real_cache_path, "wb") as f:
            pickle.dump(real_grouped, f)

    try:
        print("🔁 Loading cached syn_grouped...")
        with open(syn_cache_path, "rb") as f:
            syn_grouped = pickle.load(f)
    except Exception:
        print("Grouping synthetic dataset by class...")
        syn_grouped = group_by_class(syn_train)
        with open(syn_cache_path, "wb") as f:
            pickle.dump(syn_grouped, f)

    print(
        f"Grouping Summary\n---\nReal Classes ({len(real_grouped.keys())}): {real_grouped.keys()}"
    )
    print(f"Synthetic Classes ({len(syn_grouped.keys())}): {syn_grouped.keys()}\n\n")

    val_split = real["validation"]

    for frac in FRACTIONS:
        print(f"Processing fraction: {frac}\n---")
        create_and_push_merged_dataset(
            real_grouped=real_grouped,
            syn_grouped=syn_grouped,
            frac=frac,
            val_split=val_split,
            repo_prefix=repo_prefix,
            huggingface_token=huggingface_token,
            real_label_features=real["train"].features,
            syn_label_features=syn_train.features,
            seed=seed,
        )


if __name__ == "__main__":
    merge()
