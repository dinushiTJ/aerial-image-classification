import os
import shutil
import asyncio
from datasets import load_dataset
from PIL import Image
import click


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

async def _merge(huggingface_token: str, version: str):
    data_dest = f"dataset{version}"

    os.makedirs(data_dest, exist_ok=True)
    shutil.rmtree(data_dest)

    for cls in classes.keys():
        os.makedirs(f"{data_dest}/{cls.lower()}", exist_ok=True)

    async def process_class(cls, tok):
        ds = await asyncio.to_thread(load_dataset, f"dushj98/aerial_synthetic_{tok.lower()}{version}", token=huggingface_token)
        
        async def save_image(i, r):
            image: Image = r['image']
            await asyncio.to_thread(image.save, f"{data_dest}/{cls}/{i}.png")

        await asyncio.gather(*(save_image(i, r) for i, r in enumerate(ds["train"])))
        print(f"✅ {tok} processed.")

    # Run all dataset loading & saving concurrently
    await asyncio.gather(*(process_class(cls, tok) for cls, tok in classes.items()))

    # Load dataset after processing all images
    dataset = await asyncio.to_thread(load_dataset, "imagefolder", name=f"waikato_aerial_2017_synthetic{version}", data_dir=data_dest)
    await asyncio.to_thread(dataset.push_to_hub, f"dushj98/waikato_aerial_2017_synthetic{version}", token=huggingface_token)


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
def merge(version_tag: str, huggingface_token: str):
    asyncio.run(_merge(version=version_tag, huggingface_token=huggingface_token))


@click.group()
def cli():
    """CLI tool for partial synthetic dataset merging."""
    pass


cli.add_command(merge)


if __name__ == '__main__':
    cli()
