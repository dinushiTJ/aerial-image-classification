# CMMD calculation
# First need to 
# git clone https://github.com/sayakpaul/cmmd-pytorch.git

import numpy as np
import os
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
import click
from datasets import load_dataset
from cmmd_pytorch.main import compute_cmmd
from PIL import Image

classes = {
    'broadleaved_indigenous_hardwood': 'BIH', 
    'deciduous_hardwood': 'DHW', 
    'grose_broom': 'GBM', 
    'harvested_forest': 'HFT', 
    'herbaceous_freshwater_vege': 'HFV', 
    'high_producing_grassland': 'HPG', 
    'indigenous_forest': 'IFT', 
    'lake_pond': 'LPD', 
    'low_producing_grassland': 'LPG', 
    'manuka_kanuka': 'MKA', 
    'shortrotation_cropland': 'SRC', 
    'urban_build_up': 'UBU', 
    'urban_parkland': 'UPL'
}

# setting a seed for reproducibility
np.random.seed(0)

REAL_DS_DIR = "real"
SYNTHETIC_DS_DIR = "synthetic"

def _create_dirs(paths: str) -> None:
    """Create directories for dataset storage."""
    for path in paths:
        shutil.rmtree(path, ignore_errors=True)  # Remove existing directory if it exists
        os.makedirs(path, exist_ok=True)

        for class_name in classes.keys():
            os.makedirs(f"{path}/{class_name}", exist_ok=True)

async def save_image(img: Image, path: str):
    """Asynchronously save an image."""
    await asyncio.to_thread(img.save, path)

async def save_images_concurrently(dataset, class_name, save_dir):
    """Saves images concurrently using asyncio."""
    tasks = [
        save_image(r["image"], f"{save_dir}/{class_name}/{idx}.png")
        for idx, r in enumerate(dataset)
    ]
    await asyncio.gather(*tasks)

def _prepare_training_dataset(huggingface_token: str) -> None:
    print("Loading real dataset...")
    train_ds = load_dataset("dushj98/waikato_aerial_imagery_2017", token=huggingface_token)

    train_image_indices_for_class = {k: [] for k in classes.keys()}
    
    # Pre-index dataset to reduce redundant lookups
    train_labels = train_ds["train"]["label"]
    label_to_str = train_ds['train'].features['label'].int2str

    for idx, lbl in enumerate(train_labels):
        cls = label_to_str(lbl)
        train_image_indices_for_class[cls].append(idx)

    print("Saving real dataset images...")
    asyncio.run(
        asyncio.gather(*[
            save_images_concurrently([train_ds["train"][i] for i in imgs], cls, REAL_DS_DIR)
            for cls, imgs in train_image_indices_for_class.items()
        ])
    )

async def _prepare_datasets_async(huggingface_token: str, version: str) -> None:
    """Prepare and save datasets asynchronously."""
    random_indexes = np.random.choice(1000, 666, replace=False)

    print("Loading synthetic dataset...")
    synthetic_ds = load_dataset(f"dushj98/waikato_aerial_2017_synthetic{version}", token=huggingface_token)

    image_indices_for_class = {k: [] for k in classes.keys()}
    synth_labels = synthetic_ds["train"]["label"]
    synth_label_to_str = synthetic_ds["train"].features["label"].int2str

    for idx, lbl in enumerate(synth_labels):
        cls = synth_label_to_str(lbl)
        image_indices_for_class[cls].append(idx)

    print("Saving synthetic dataset images...")

    tasks = [
        save_images_concurrently(
            [synthetic_ds["train"][image_indices_for_class[cls][i]]
             for i in random_indexes if i < len(image_indices_for_class[cls])],
            cls,
            f"{SYNTHETIC_DS_DIR}{version}"
        )
        for cls in classes.keys()
    ]

    await asyncio.gather(*tasks)

def _prepare_datasets(huggingface_token: str, version: str) -> None:
    """Prepare and save datasets asynchronously."""

    asyncio.run(_prepare_datasets_async(huggingface_token, version))

@click.command()
@click.option(
    '--version-tag', '-v',
    type=str,
    required=True,
    help='Version tag suffix of the synthetic datasets'
)
@click.option(
    '--huggingface-token', '-h',
    type=str,
    required=True,
    help='Huggingface token'
)
@click.option(
    '--dataset-prep', '-d',
    is_flag=True,
    default=False,
    help='Whether or not to prepare the dataset before the CMMD calculation'
)
def cmmd(version_tag: str, huggingface_token: str, dataset_prep: bool):
    """Main CLI function for computing CMMD."""

    # _prepare_training_dataset(huggingface_token=huggingface_token)

    if dataset_prep:
        _create_dirs(paths=[f"{SYNTHETIC_DS_DIR}{version_tag}"])
        _prepare_datasets(huggingface_token=huggingface_token, version=version_tag)
    
    else:
        print("Computing CMMD scores...")
        for cls in classes.keys():
            cmmd_score = compute_cmmd(f"{REAL_DS_DIR}/{cls}", f"{SYNTHETIC_DS_DIR}{version_tag}/{cls}")
            click.echo(f"✅ CMMD for {cls} is: {cmmd_score}")

@click.group()
def cli():
    """CLI entrypoint for calculating CMMD."""
    pass

cli.add_command(cmmd)

if __name__ == '__main__':
    cli()
