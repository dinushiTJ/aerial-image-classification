# Pulls hf dataset and saves to disk.

import os
import shutil
import asyncio
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


async def _save_dataset(version: str, huggingface_token: str) -> None:
    data_dir = "saved_datasets"
    dataset_name = f"dushj98/waikato_aerial_2017_synthetic{version}"

    train_ds = load_dataset(dataset_name, token=huggingface_token)
    dataset_dir_name = dataset_name.split("/")[-1]
    dataset_dir = os.path.join(data_dir, dataset_dir_name)

    os.makedirs(dataset_dir, exist_ok=True)
    shutil.rmtree(dataset_dir)

    for class_name in classes.keys():
        os.makedirs(f"{dataset_dir}/{class_name}", exist_ok=True)
                
    image_indices_for_class = {k: [] for k in classes.keys()}
    train_labels = train_ds["train"]["label"]
    label_to_str = train_ds['train'].features['label'].int2str

    for idx, lbl in enumerate(train_labels):
        cls = label_to_str(lbl)
        image_indices_for_class[cls].append(idx)

    click.echo(f"Saving images. from synthetic{version}...")
    await asyncio.gather(*[
        _save_images_concurrently([train_ds["train"][i] for i in imgs], cls_, dataset_dir)
        for cls_, imgs in image_indices_for_class.items()
    ])


@click.group()
def cli():
    """CLI entrypoint for calculating CMMD."""
    pass


@cli.command()
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
def save(version_tag: str, huggingface_token: str):
    """Prepares the synthetic images"""
    try:
        asyncio.run(_save_dataset(version=version_tag, huggingface_token=huggingface_token))
        click.echo("✅ Done!")
    except DatasetNotFoundError:
        click.echo("The dataset does not exist on the huggingface hub", err=True)


if __name__ == '__main__':
    cli()
