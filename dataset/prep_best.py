# Prepare best datasets 
# by combining classes with 
# lowest CMMD & FID.

import json
from math import isnan
import json
import os
import asyncio
import click
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from PIL import Image
from typing import Literal, Type


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

json_files = [
    "synthetic_ti_v1_res_cls",
    "synthetic_ti_v1_upscaled_res_cls",
    "synthetic_ti_v2_res_cls",
    "synthetic_ti_v2_upscaled_res_cls",
    "synthetic_v0_res_cls",
    "synthetic_v0_upscaled_res_cls",
    "synthetic_v1_res_cls",
    "synthetic_v1_upscaled_res_cls",
    "synthetic_v2_res_cls",
    "synthetic_v2_upscaled_res_cls"
]

Cls = Type[Literal["bih", "dhw", "gbm", "hft", "hfv", "hpg", "ift", "lpd", "lpg", "mka", "src", "ubu", "upl"]]
Metric = Type[Literal["cmmd", "fid"]]
SYNTHETIC_DS_DIR = "synthetic"


async def _pull_cls_images(metric: Metric, class_name: Cls, dataset: str, huggingface_token: str) -> None:
    dataset_name = f"dushj98/waikato_aerial_2017_synthetic_{dataset}".replace("_up", "_upscaled")
    synthetic_ds = load_dataset(dataset_name, token=huggingface_token)

    dir_with_version = f"{SYNTHETIC_DS_DIR}_best_{metric}"
    os.makedirs(dir_with_version, exist_ok=True)
    os.makedirs(os.path.join(dir_with_version, class_name), exist_ok=True)

    synth_labels = synthetic_ds["train"]["label"]
    synth_label_to_str = synthetic_ds["train"].features["label"].int2str

    image_indices_for_class = []

    for idx, lbl in enumerate(synth_labels):
        cls = synth_label_to_str(lbl)
        if cls == class_name:
            image_indices_for_class.append(idx)
    
    for idx, index in enumerate(image_indices_for_class):
        row = synthetic_ds["train"][index]
        image_ = row["image"]
        image_.save(f"{dir_with_version}/{class_name}/{idx}.png")


async def _prepare_dataset_with_most_sim_cls(huggingface_token: str) -> None:
    base_dir = "/home/dj191/research/code/similarity"

    # Dictionary to store data for each class
    class_data = {}

    # Process files in pairs (CMMD and FID)
    for f in json_files:
        fid_file = f"{f}_fid.json"
        cmmd_file = f"{f}.json"

        fid_path = os.path.join(base_dir, fid_file)
        cmmd_path = os.path.join(base_dir, cmmd_file)

        with open(fid_path, 'r') as f:
            fid_data = json.load(f)

        with open(cmmd_path, 'r') as f:
            cmmd_data = json.load(f)

        dataset_name = fid_file.replace("synthetic_", "").replace("_res_cls_fid.json", "")
        dataset_name = dataset_name.replace("_upscaled", "_up")

        for class_name in fid_data:
            fid_score = fid_data[class_name].get("fid")
            cmmd_score = cmmd_data[class_name].get("cmmd")

            if class_name not in class_data:
                class_data[class_name] = []

            class_data[class_name].append({
                "Dataset": dataset_name,
                "CMMD": cmmd_score if cmmd_score is not None else float('nan'),
                "FID": fid_score if fid_score is not None else float('nan')
            })
    

    for class_name, datasets in class_data.items():
        min_cmmd = min(datasets, key=lambda d: d["CMMD"] if not isnan(d["CMMD"]) else float('inf'))
        min_fid = min(datasets, key=lambda d: d["FID"] if not isnan(d["FID"]) else float('inf'))

        tasks = [
            _pull_cls_images(metric="cmmd", class_name=class_name.lower(), dataset=min_cmmd["Dataset"], huggingface_token=huggingface_token),
            _pull_cls_images(metric="fid", class_name=class_name.lower(), dataset=min_fid["Dataset"], huggingface_token=huggingface_token),
        ]
        await asyncio.gather(*tasks)
    
    for m in ["cmmd", "fid"]:
        click.echo(f"Uploading best {m} dataset to hf...")
        best_ds = await asyncio.to_thread(load_dataset, "imagefolder", name=f"waikato_aerial_2017_synthetic_best_{m}", data_dir=f"{SYNTHETIC_DS_DIR}_best_{m}")
        await asyncio.to_thread(best_ds.push_to_hub, f"dushj98/waikato_aerial_2017_synthetic_best_{m}", token=huggingface_token)


@click.group()
def cli():
    """CLI entrypoint for preping most similar datasets."""
    pass


@cli.command()
@click.option(
    '--huggingface-token', '-h',
    type=str,
    required=True,
    help='Huggingface token'
)
def prep(huggingface_token: str):
    """Prepares the real images"""
    try:
        asyncio.run(_prepare_dataset_with_most_sim_cls(huggingface_token=huggingface_token))
        click.echo("✅ Done!")
    except DatasetNotFoundError:
        click.echo("The dataset does not exist on the huggingface hub", err=True)


if __name__ == '__main__':
    cli()
