# Image upscaling
# model: stabilityai/stable-diffusion-x4-upscaler

import click
from datasets import load_dataset
from datasets.exceptions import DatasetNotFoundError
from diffusers import StableDiffusionUpscalePipeline
import torch


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

model_id = "stabilityai/stable-diffusion-x4-upscaler"
pipeline = StableDiffusionUpscalePipeline.from_pretrained(model_id, torch_dtype=torch.float16)
pipeline = pipeline.to("cuda")


def _upscale_image(r, int2str):
    image = r["image"]
    label = r["label"]
    label_str = int2str(label)

    low_res_image = image.resize((128, 128))
    label_parts = label_str.split("_")
    prompt = f"A real aerial view of {' '.join(label_parts)} area in Waikato, New Zealand"

    upscaled_image = pipeline(prompt=prompt, image=low_res_image).images[0]
    r["image"] = upscaled_image
    
    return r


def _upscale(dataset_name: str, version: str, huggingface_token: str) -> None:
    dataset_name = f"{dataset_name}{version}"
    synthetic_ds = load_dataset(dataset_name, token=huggingface_token)

    upscaled_ds = synthetic_ds.map(_upscale_image, fn_kwargs={"int2str": synthetic_ds["train"].features["label"].int2str})
    upscaled_ds.push_to_hub(f"{dataset_name}_upscaled", token=huggingface_token)


@click.group()
def cli():
    """CLI entrypoint for Image upscaling."""
    pass


@cli.command()
@click.option(
    '--dataset-name', '-d',
    type=str,
    default="dushj98/waikato_aerial_2017_synthetic",
    required=True,
    help='Name of the huggingface dataset to be upscaled.'
)
@click.option(
    '--version-tag', '-v',
    type=str,
    required=True,
    help='Version tag suffix of the huggingface datasets. Eg: _v0, _v1, etc.'
)
@click.option(
    '--huggingface-token', '-h',
    type=str,
    required=True,
    help='Huggingface token'
)
def upscale(dataset_name: str, version_tag: str, huggingface_token: str):
    """Upscales 512x512 images and outputs images with the same resolution"""
    try:
        _upscale(dataset_name=dataset_name, version=version_tag, huggingface_token=huggingface_token)
        click.echo("✅ Done!")
    except DatasetNotFoundError:
        click.echo("The dataset does not exist on the huggingface hub", err=True)
    click.echo("✅ Done!")


if __name__ == '__main__':
    cli()
