# FID & CMMD calculation
# First need to 
# git clone https://github.com/sayakpaul/cmmd-pytorch.git
# for CMMD and the FID calculation is done using torcheval lib.

import numpy as np
import json
import os
import shutil
import asyncio
from concurrent.futures import ThreadPoolExecutor
import click
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from PIL import Image

from cmmd_pytorch.main import compute_cmmd
from fid_pytorch.main import compute_fid


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


async def _save_image(img: Image, path: str):
    """Asynchronously saves an image."""
    await asyncio.to_thread(img.save, path)


async def _save_images_concurrently(dataset, class_name, save_dir):
    """Saves images concurrently using asyncio."""
    tasks = [
        _save_image(r["image"], f"{save_dir}/{class_name}/{idx}.png")
        for idx, r in enumerate(dataset)
    ]
    await asyncio.gather(*tasks)


async def _prepare_all_dir(src_dir: str, dst_dir: str) -> None:
    os.makedirs(dst_dir, exist_ok=True)

    class_dirs = [d for d in os.listdir(src_dir) if os.path.isdir(os.path.join(src_dir, d))]
    for d in class_dirs:
        class_images = [f for f in os.listdir(os.path.join(src_dir, d)) if f.endswith(".png")]
        for img in class_images:
          shutil.copy(src=os.path.join(src_dir, d, img), dst=os.path.join(dst_dir, f"{d}_{img}"))


async def _prepare_real_dataset(huggingface_token: str, force_prep: bool = False) -> None:
    """Preprares the real dataset"""
    dataset_name = "dushj98/waikato_aerial_imagery_2017"
    train_ds = load_dataset(dataset_name, token=huggingface_token)

    os.makedirs(REAL_DS_DIR, exist_ok=True)

    class_dirs = [d for d in os.listdir(REAL_DS_DIR) if os.path.isdir(os.path.join(REAL_DS_DIR, d))]
    
    images = list()
    for d in class_dirs:
        class_images = [f for f in os.listdir(os.path.join(REAL_DS_DIR, d)) if f.endswith(".png")]
        images.extend(class_images)
      
    # click.echo(len(images))
    
    if len(images) != 8658 or force_prep:
        click.echo("Preparing the real dataset...")

        shutil.rmtree(REAL_DS_DIR)

        for class_name in classes.keys():
            os.makedirs(f"{REAL_DS_DIR}/{class_name}", exist_ok=True)
                
        train_image_indices_for_class = {k: [] for k in classes.keys()}
    
        train_labels = train_ds["train"]["label"]
        label_to_str = train_ds['train'].features['label'].int2str

        for idx, lbl in enumerate(train_labels):
            cls = label_to_str(lbl)
            train_image_indices_for_class[cls].append(idx)

        click.echo("Saving real dataset images...")
        await asyncio.gather(*[
            _save_images_concurrently([train_ds["train"][i] for i in imgs], cls, REAL_DS_DIR)
            for cls, imgs in train_image_indices_for_class.items()
        ])
    
    # preping for all images
    click.echo("Saving real_all dataset images...")
    await _prepare_all_dir(src_dir=REAL_DS_DIR, dst_dir=f"{REAL_DS_DIR}_all")


async def _prepare_synthetic_dataset(version: str, huggingface_token: str, force_prep: bool = False) -> None:
    dataset_name = f"dushj98/waikato_aerial_2017_synthetic{version}"
    synthetic_ds = load_dataset(dataset_name, token=huggingface_token)

    dir_with_version = f"{SYNTHETIC_DS_DIR}{version}"
    os.makedirs(dir_with_version, exist_ok=True)

    class_dirs = [d for d in os.listdir(dir_with_version) if os.path.isdir(os.path.join(dir_with_version, d))]
    
    images = list()
    for d in class_dirs:
        class_images = [f for f in os.listdir(os.path.join(dir_with_version, d)) if f.endswith(".png")]
        images.extend(class_images)
      
    # click.echo(len(images))

    if len(images) != 8658 or force_prep:
        click.echo(f"Preparing the synthetic{version} dataset...")

        random_indices = np.random.choice(1000, 666, replace=False)

        image_indices_for_class = {k: [] for k in classes.keys()}
        synth_labels = synthetic_ds["train"]["label"]
        synth_label_to_str = synthetic_ds["train"].features["label"].int2str

        for idx, lbl in enumerate(synth_labels):
            cls = synth_label_to_str(lbl)
            image_indices_for_class[cls].append(idx)
        
        max_rand_index = max(random_indices)
        for cls, cls_indices in image_indices_for_class.items():
            if max_rand_index > len(cls_indices) - 1:
                click.echo(
                    f"Random image selection failed due to the {cls} class not having 1000 images.", 
                    err=True
                )
                return

        shutil.rmtree(dir_with_version)

        for class_name in classes.keys():
            os.makedirs(f"{dir_with_version}/{class_name}", exist_ok=True)

        click.echo("Saving synthetic dataset images...")
        tasks = [
            _save_images_concurrently(
                [synthetic_ds["train"][image_indices_for_class[cls][i]] for i in random_indices if i < len(image_indices_for_class[cls])],
                cls,
                f"{SYNTHETIC_DS_DIR}{version}"
            )
            for cls in classes.keys()
        ]
        await asyncio.gather(*tasks)
    
    # preping for all images
    click.echo("Saving real_all dataset images...")
    await _prepare_all_dir(src_dir=dir_with_version, dst_dir=f"{dir_with_version}_all")


@click.group()
def cli():
    """CLI entrypoint for calculating CMMD."""
    pass


@cli.group()
def prep():
    pass


@prep.command()
@click.option(
    '--huggingface-token', '-h',
    type=str,
    required=True,
    help='Huggingface token'
)
@click.option(
    '--force-prep', '-f',
    is_flag=True,
    default=False,
    help='If present, recreates the dataset if it already exists.'
)
def real(huggingface_token: str, force_prep: bool):
    """Prepares the real images"""
    try:
        asyncio.run(_prepare_real_dataset(huggingface_token=huggingface_token, force_prep=force_prep))
        click.echo("✅ Done!")
    except DatasetNotFoundError:
        click.echo("The dataset does not exist on the huggingface hub", err=True)


@prep.command()
@click.option(
    '--version-tag', '-v',
    type=str,
    required=True,
    help='Version tag suffix of the synthetic datasets. Eg: _v0, _v1, etc.'
)
@click.option(
    '--huggingface-token', '-h',
    type=str,
    required=True,
    help='Huggingface token'
)
@click.option(
    '--force-prep', '-f',
    is_flag=True,
    default=False,
    help='If present, recreates the dataset if it already exists.'
)
def synthetic(version_tag: str, huggingface_token: str, force_prep: bool):
    """Prepares the synthetic images"""
    try:
        asyncio.run(_prepare_synthetic_dataset(version=version_tag, huggingface_token=huggingface_token, force_prep=force_prep))
        click.echo("✅ Done!")
    except DatasetNotFoundError:
        click.echo("The dataset does not exist on the huggingface hub", err=True)


@cli.command()
@click.option(
    '--version-tag', '-v',
    type=str,
    required=True,
    help='Version tag suffix of the synthetic datasets'
)
@click.option(
    '--cls', '-c',
    is_flag=True,
    default=False,
    help='If present, class-wise CMMD will be calculated.'
)
@click.option(
    '--dataset', '-d',
    is_flag=True,
    default=False,
    help='If present, dataset-wide CMMD will be calculated.'
)
@click.option(
    '--all', '-a',
    is_flag=True,
    default=False,
    help='If present, both class-wise and dataset-wide CMMD will be calculated.'
)
def cmmd(version_tag: str, cls: bool, dataset: bool, all: bool):
    """Calculates CMMD values for two datasets"""

    if cls or all:
        click.echo(f"Calculating class-wise CMMD values for synthetic{version_tag}...")
        class_wise_cmmd_values = {k: {"token": v, "cmmd": None} for k, v in classes.items()}
        
        for cls in classes.keys():
            cmmd_score = compute_cmmd(f"{REAL_DS_DIR}/{cls}", f"{SYNTHETIC_DS_DIR}{version_tag}/{cls}")
            click.echo(f"✅ CMMD for {cls} is: {cmmd_score}")
            class_wise_cmmd_values[cls]["cmmd"] = float(cmmd_score)
        
        with open(f"synthetic{version_tag}_res_cls.json", mode="w") as f:
          json.dump(class_wise_cmmd_values, f, indent=4)
    
    if dataset or all:
        click.echo(f"Calculating dataset-wide CMMD value for synthetic{version_tag}...")
        dataset_cmmd_score = compute_cmmd(f"{REAL_DS_DIR}_all", f"{SYNTHETIC_DS_DIR}{version_tag}_all")
        click.echo(f"✅ CMMD for synthetic{version_tag}: {dataset_cmmd_score}")
        
        with open(f"synthetic{version_tag}_res_ds.json", mode="w") as f:
          json.dump({"dataset_cmmd_score": float(dataset_cmmd_score)}, f, indent=4)
    
    click.echo("✅ Done!")


@cli.command()
@click.option(
    '--version-tag', '-v',
    type=str,
    required=True,
    help='Version tag suffix of the synthetic datasets'
)
@click.option(
    '--cls', '-c',
    is_flag=True,
    default=False,
    help='If present, class-wise FID will be calculated.'
)
@click.option(
    '--dataset', '-d',
    is_flag=True,
    default=False,
    help='If present, dataset-wide FID will be calculated.'
)
@click.option(
    '--all', '-a',
    is_flag=True,
    default=False,
    help='If present, both class-wise and dataset-wide FID will be calculated.'
)
def fid(version_tag: str, cls: bool, dataset: bool, all: bool):
    """Calculates FID values for two datasets"""

    if cls or all:
        click.echo(f"Calculating class-wise FID values for synthetic{version_tag}...")
        class_wise_fid_values = {k: {"token": v, "fid": None} for k, v in classes.items()}
        
        for cls in classes.keys():
            fid_score = compute_fid(f"{REAL_DS_DIR}/{cls}", f"{SYNTHETIC_DS_DIR}{version_tag}/{cls}")
            click.echo(f"✅ FID for {cls} is: {fid_score}")
            class_wise_fid_values[cls]["fid"] = float(fid_score)
        
        with open(f"synthetic{version_tag}_res_cls_fid.json", mode="w") as f:
          json.dump(class_wise_fid_values, f, indent=2)
    
    if dataset or all:
        click.echo(f"Calculating dataset-wide FID value for synthetic{version_tag}...")
        dataset_fid_score = compute_fid(f"{REAL_DS_DIR}_all", f"{SYNTHETIC_DS_DIR}{version_tag}_all")
        click.echo(f"✅ FID for synthetic{version_tag}: {dataset_fid_score}")
        
        with open(f"synthetic{version_tag}_res_ds_fid.json", mode="w") as f:
          json.dump({"dataset_fid_score": float(dataset_fid_score)}, f, indent=2)
    
    click.echo("✅ Done!")


if __name__ == '__main__':
    cli()
