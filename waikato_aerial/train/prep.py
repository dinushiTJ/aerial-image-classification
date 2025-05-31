# Prep training dataset for textual-inversion

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

DS_DIR = "dataset"


async def _save_image(img: Image, path: str):
    """Asynchronously saves an image."""
    await asyncio.to_thread(img.save, path)


async def _prepare_dataset(huggingface_token: str, force_prep: bool = False) -> None:
    dataset_name = f"dushj98/waikato_aerial_2017_sd_ft"
    ds = load_dataset(dataset_name, token=huggingface_token)
    os.makedirs(DS_DIR, exist_ok=True)
    
    class_dirs = [d for d in os.listdir(DS_DIR) if os.path.isdir(os.path.join(DS_DIR, d))]
    
    images = list()
    for d in class_dirs:
        class_images = [f for f in os.listdir(os.path.join(DS_DIR, d)) if f.endswith(".png")]
        images.extend(class_images)
    
    click.echo(f"Images: {len(images)}\nDataset: {len(ds)}")

    if len(images) != 260 or force_prep:
        shutil.rmtree(DS_DIR)
        for k in classes.keys():
            os.makedirs(os.path.join(DS_DIR, k), exist_ok=True)

        click.echo(f"Preparing the training dataset...")

        label_to_str = ds["train"].features["label"].int2str
        click.echo("Saving images...")
        
        cls_images = {k: [] for k in classes.keys()}
        for r in ds["train"]:
            cls = label_to_str(r["label"])
            cls_images[cls].append(r["image"])
        
        
        tasks = []
        for cls, images in cls_images.items():
            for idx, img in enumerate(images):
                tasks.append(_save_image(img, f"{DS_DIR}/{cls}/{idx}.png"))
            
        await asyncio.gather(*tasks)


@click.group()
def cli():
    """CLI entrypoint for calculating CMMD."""
    pass


@cli.command()
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
def prep(huggingface_token: str, force_prep: bool):
    """Prepares the training dataset"""
    try:
        asyncio.run(_prepare_dataset(huggingface_token=huggingface_token, force_prep=force_prep))
        click.echo("✅ Done!")
    except DatasetNotFoundError:
        click.echo("The dataset does not exist on the huggingface hub", err=True)


@cli.command()
def prep_model_dir():
    """Prepares the model dirs"""
    try:
        for k, v in classes.items():
            os.makedirs(f"ti_models/{v.lower()}_textual_inversion", exist_ok=True)
            os.makedirs(f"lora_models/{v.lower()}_lora", exist_ok=True)

        click.echo("✅ Done!")
    except DatasetNotFoundError:
        click.echo("The dataset does not exist on the huggingface hub", err=True)


if __name__ == '__main__':
    cli()
