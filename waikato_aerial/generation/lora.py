# Generating Dreambooth + LoRA Images

import os
import shutil
from diffusers import StableDiffusionPipeline
import torch
from datasets import load_dataset
import click
from typing import Literal
import time


prompts = {
  "BIH": [
    "A top-down <BIH> view of a dense forest with broadleaved native trees.",
    "A drone shot of <BIH> showing a forest with large, green leaves and a shaded understory.",
    "An aerial <BIH> view of a native hardwood forest with sunlight filtering through the canopy.",
    "A top-down <BIH> image of a forest with tall, broadleaved trees and a mossy floor.",
    "A drone view of <BIH> featuring a forest with a mix of native hardwood trees.",
    "An aerial <BIH> shot of a forest with a small stream and broadleaved trees.",
    "A top-down <BIH> scene of a native forest with a dirt trail and dense vegetation.",
    "A drone view of <BIH> showing a forest with large, green leaves.",
    "An aerial <BIH> image of a forest with a mix of broadleaved and coniferous trees.",
    "A top-down <BIH> view of a forest with a small pond and native hardwood trees.",
    "A drone shot of <BIH> featuring a forest with a rocky outcrop and broadleaved trees.",
    "An aerial <BIH> view of a forest with a dirt path and dense, green foliage.",
    "A top-down <BIH> image of a forest with a river and native hardwood trees.",
    "A drone view of <BIH> showing a forest with a small clearing and broadleaved trees.",
    "An aerial <BIH> shot of a forest with a distant hill and native hardwood trees.",
    "A top-down <BIH> scene of a forest with a small waterfall and broadleaved trees.",
    "A drone view of <BIH> featuring a forest with a mix of broadleaved and deciduous trees.",
    "An aerial <BIH> image of a forest with a small pond and native hardwood trees.",
    "A top-down <BIH> view of a forest with a dirt road and dense, green foliage.",
    "A <BIH> aerial view of a broadleaved indigenous hardwood trees"
  ],
  "DHW": [
    "A top-down <DHW> view of a forest with deciduous trees in autumn.",
    "A drone shot of <DHW> showing a forest with deciduous trees.",
    "A top-down <DHW> image of a forest with bare trees.",
    "A drone view of <DHW> featuring a forest with a mix of deciduous and evergreen trees.",
    "An aerial <DHW> shot of a forest with a small stream.",
    "A top-down <DHW> scene of a deciduous forest with a dirt trail.",
    "A drone view of <DHW> vast deciduous forest.",
    "An aerial <DHW> image of a forest with a mix of deciduous and coniferous trees.",
    "A top-down <DHW> view of a forest with a small pond.",
    "A drone shot of <DHW> featuring a forest with a rocky outcrop and deciduous trees.",
    "An aerial <DHW> view of a forest with a dirt path.",
    "A top-down <DHW> image of a forest with a river and deciduous trees.",
    "A drone view of <DHW> showing a forest with a small clearing.",
    "An aerial <DHW> shot of a forest with a distant hill and deciduous trees.",
    "A top-down <DHW> scene of a deciduous forest alongside a road.",
    "A drone view of <DHW> featuring a forest with a mix of deciduous and evergreen trees.",
    "An aerial <DHW> image of a forest with a small water source.",
    "A top-down <DHW> view of a forest with a dirt road.",
    "A drone shot of <DHW> showing a forest with a small stream and deciduous trees.",
    "A <DHW> aerial view of a forest with a road on a side."
  ],
  "GBM": [
    "A top-down <GBM> view of a field covered in dense gorse and broom bushes.",
    "A drone shot of <GBM> showing a sloped area with scattered gorse and broom plants.",
    "An aerial <GBM> view of a field with a thick cover of gorse and broom bushes.",
    "A top-down <GBM> image of a landscape filled with yellow gorse flowers and green broom.",
    "A drone view of <GBM> capturing a field with patches of gorse and broom bushes.",
    "An aerial <GBM> shot of an area with a mix of gorse and broom along a dirt path.",
    "A top-down <GBM> scene showing a field with gorse and broom bushes beside a small stream.",
    "A drone view of <GBM> displaying an open field with dense gorse and broom along a winding path.",
    "An aerial <GBM> image of a flat field with heavy gorse and broom coverage and no trees in sight.",
    "A top-down <GBM> view of a section of land with sparse gorse and broom and a dirt road running through it.",
    "A drone shot of <GBM> showing a meadow with a mix of gorse and broom bushes and a small pond.",
    "An aerial <GBM> view of an area with a blanket of gorse and broom, and visible dirt paths.",
    "A top-down <GBM> image of a flat area with scattered gorse and broom bushes and no water features.",
    "A drone view of <GBM> showing an expanse of land with dense patches of gorse and broom and a dirt trail.",
    "An aerial <GBM> shot of a field with clusters of gorse and broom bushes and a flowing stream.",
    "A top-down <GBM> scene of an area covered in thick gorse and broom, with visible terrain features.",
    "A drone view of <GBM> showcasing a field with gorse and broom plants covering the landscape.",
    "An aerial <GBM> image of an open space with sparse gorse and broom and a dirt path running through it.",
    "A <GBM> aerial view of dense gorse and broom",
    "A drone shot of <GBM> showing an expanse of land with dense gorse and broom, bordered by dirt paths."
  ],
  "HFT": [
    "A top-down <HFT> view of a cleared forest with tree stumps and logging tracks.",
    "A drone shot of <HFT> showing a harvested forest.",
    "An aerial <HFT> view of a recently logged area",
    "A top-down <HFT> image of a harvested forest with regrowth starting to appear.",
    "A drone view of <HFT> featuring a cleared forest with dirt roads.",
    "An aerial <HFT> shot of a harvested forest with a stream running through it.",
    "A top-down <HFT> scene of a logged area with scattered debris and new seedlings.",
    "A drone view of <HFT> showing a harvested forest.",
    "An aerial <HFT> image of a cleared forest with patches of regrowth.",
    "A top-down <HFT> view of a harvested forest with a logging truck on a dirt road.",
    "A drone shot of <HFT> featuring a logged area with scattered stumps.",
    "An aerial <HFT> view of a harvested forest with a small pond and regrowth.",
    "A top-down <HFT> image of a cleared forest with a river running through it.",
    "A drone view of <HFT> showing a harvested forest.",
    "An aerial <HFT> shot of a logged area with a dirt road and scattered debris.",
    "A top-down <HFT> scene of a harvested forest and regrowth.",
    "A drone view of <HFT> featuring a cleared forest with scattered stumps and regrowth.",
    "An aerial <HFT> image of a harvested forest with a logging camp.",
    "A top-down <HFT> view of a logged area with scattered branches and new seedlings.",
    "A <HFT> aerial view of a halfway harvested forest"
  ],
  "HFV": [
    "A top-down <HFV> view of a wetland with lush green vegetation and a small pond.",
    "A drone shot of <HFV> showing a freshwater marsh with tall, waving grasses and a winding stream.",
    "An aerial <HFV> view of a wetland with patches of soft, green plants and a small water body.",
    "A top-down <HFV> image of a freshwater marsh with dense grasses and a clear stream running through it.",
    "A drone view of <HFV> capturing a wetland with soft green vegetation surrounding a pond.",
    "An aerial <HFV> shot of a freshwater marsh with tall reeds and a meandering creek.",
    "A top-down <HFV> scene of a wetland with vibrant green plants and a stream cutting through.",
    "A drone view of <HFV> showing a freshwater marsh dotted with small ponds and tall grasses.",
    "An aerial <HFV> image of a wetland with dense green vegetation and winding water paths.",
    "A top-down <HFV> view of a lush marsh with a network of small streams and green vegetation.",
    "A drone shot of <HFV> featuring a wetland with dense reeds and a tranquil pond.",
    "An aerial <HFV> view of a freshwater marsh with varying shades of green vegetation and a creek.",
    "A top-down <HFV> image of a wetland with rich greenery and scattered small water bodies.",
    "A drone view of <HFV> showing a freshwater marsh with tall grasses and a small, winding stream.",
    "An aerial <HFV> shot of a wetland with a mix of soft green plants and clear water paths.",
    "A top-down <HFV> scene of a wetland with lush vegetation and multiple small ponds.",
    "A drone view of <HFV> featuring a wetland with patches of soft green plants and a small stream.",
    "An aerial <HFV> image of a freshwater marsh with dense vegetation and winding watercourses.",
    "A top-down <HFV> view of a wetland with thick, soft vegetation and small water pools scattered across.",
    "A <HFV> aerial view of a herbaceous freshwater vege area"
  ],
  "HPG": [
    "A top-down <HPG> view of a well-maintained grassland with dense, even growth.",
    "A drone shot of <HPG> showing a large, uniform field with tall grass.",
    "An aerial <HPG> view of a high-yielding grassland with a dirt road cutting through.",
    "A top-down <HPG> image capturing a productive grassland with a central irrigation system.",
    "A drone view of <HPG> featuring a thick, healthy grass cover across a vast area.",
    "An aerial <HPG> shot showing a well managed grass field.",
    "A top-down <HPG> scene highlighting a structured grassland.",
    "A drone view of <HPG> capturing a field with rich, evenly distributed grass.",
    "An aerial <HPG> image showing an expansive, high-production grassland with irrigation ditches.",
    "A top-down <HPG> view of a high-yield grass field.",
    "A drone shot of <HPG> displaying a maintained grassland with uniform plant height.",
    "An aerial <HPG> perspective showing a productive pasture with sections of similar growth.",
    "A top-down <HPG> image of a grassland optimized for maximum yield.",
    "A drone view of <HPG> capturing a well-irrigated, thriving field.",
    "An aerial <HPG> shot featuring a high-production field with clear pathways.",
    "A top-down <HPG> view displaying organized rows of lush grass.",
    "A drone shot of <HPG> showcasing a vast, carefully managed grassland.",
    "An aerial <HPG> image of a productive field of glass.",
    "A top-down <HPG> scene featuring a grassland",
    "A <HPG> aerial view with a grassland with dense, even growth."
  ],
  "IFT": [
    "A top-down <IFT> view of a dense native forest with tall trees.",
    "A drone shot of <IFT> showing a large forest with native trees and thick vegetation.",
    "An aerial <IFT> view of a native forest with extensive tree cover and dense foliage.",
    "A top-down <IFT> image of a native forest with dense, unbroken tree canopy.",
    "A drone view of <IFT> featuring a forest filled with native trees and dense undergrowth.",
    "An aerial <IFT> shot of a forest with dense native trees and rich vegetation.",
    "A top-down <IFT> scene of a large native forest with dense tree coverage.",
    "A drone view of <IFT> showing a forest with thick, continuous tree canopy.",
    "An aerial <IFT> image of a vast native forest with a dense and undisturbed tree cover.",
    "A top-down <IFT> view of a dense native forest with an unbroken tree canopy.",
    "A drone shot of <IFT> featuring a forest with thick native tree coverage.",
    "An aerial <IFT> view of a large native forest with dense tree cover and vegetation.",
    "A top-down <IFT> image of a forest with dense, uninterrupted native trees.",
    "A drone view of <IFT> showing a vast forest with continuous native tree cover.",
    "An aerial <IFT> shot of a dense native forest with abundant tree coverage.",
    "A top-down <IFT> scene of a thick native forest with uninterrupted tree canopy.",
    "A drone view of <IFT> featuring a large, dense forest with native trees.",
    "An aerial <IFT> image of a dense native forest with rich, continuous vegetation.",
    "A top-down <IFT> view of a vast native forest with uninterrupted tree canopy.",
    "A <IFT> aerial view of an indigenous forest."
  ],
  "LPD": [
    "A top-down <LPD> view of a tranquil lake surrounded by dense forest.",
    "A drone shot of <LPD> showing a calm pond with dense vegetation around it.",
    "An aerial <LPD> view of a large lake with a small island and encircling trees.",
    "A top-down <LPD> image of a pond with clusters of reeds and vegetation.",
    "A drone view of <LPD> featuring a lake with trees lining its shores.",
    "An aerial <LPD> shot of a pond with dense vegetation and surrounding plant life.",
    "A top-down <LPD> scene of a lake with surrounding hills and forest cover.",
    "A drone view of <LPD> showing a pond with dense forest and scattered vegetation.",
    "An aerial <LPD> image of a lake surrounded by a thick forest and shoreline vegetation.",
    "A top-down <LPD> view of a pond with lush vegetation surrounding the water.",
    "A drone shot of <LPD> showing a calm lake with trees and greenery along its edges.",
    "An aerial <LPD> view of a pond with surrounding wetland vegetation and forest.",
    "A top-down <LPD> image of a lake with a forested area along its banks.",
    "A drone view of <LPD> showing a pond surrounded by wild vegetation and trees.",
    "An aerial <LPD> shot of a lake with an island and surrounding dense forest.",
    "A top-down <LPD> scene of a pond with dense greenery around the water's edge.",
    "A drone view of <LPD> featuring a lake with abundant trees and wetland plants around it.",
    "An aerial <LPD> image of a pond surrounded by lush vegetation and trees.",
    "A top-down <LPD> view of a lake with a forest area surrounding the water.",
    "A <LPD> aerial view where a large lake is on a side"
  ],
  "LPG": [
    "A top-down <LPG> view of patchy, dry grassland with scattered shrubs.",
    "A drone shot of <LPG> showing thin, sparse grass covering a rolling hillside.",
    "An aerial <LPG> view of a grassland with rocky outcrops and little vegetation.",
    "A top-down <LPG> image of a dry, low-producing field with cracked soil.",
    "A drone view of <LPG> featuring a grassland with scattered weeds and bare patches.",
    "An aerial <LPG> shot of a low production grassland.",
    "A top-down <LPG> scene of a low-producing field with scattered wildflowers.",
    "A drone view of <LPG> showing a grassland with a dirt path running through it.",
    "An aerial <LPG> image of a dry unhealthy grassland.",
    "A top-down <LPG> view of a patchy and rocky grassland with a small pond and sparse vegetation.",
    "A drone shot of <LPG> featuring a low-producing field with scattered rocks.",
    "An aerial <LPG> view of a low-production grassland.",
    "A top-down <LPG> image of a dry, barren grassland with no trees.",
    "A drone view of <LPG> showing a grassland with a small stream and sparse grass.",
    "An aerial <LPG> shot of a low-producing field with scattered trees.",
    "A top-down <LPG> scene of a grassland with a dirt road, scattered weeds, and patches.",
    "A drone view of <LPG> featuring a grassland with a few scattered trees.",
    "An aerial <LPG> image of a dry, low-producing field with a farmhouse in the distance.",
    "A top-down <LPG> view of a grassland with a sparse vegetation.",
    "A <LPG> aerial view of a patchy low-production grass area"
  ],
  "MKA": [
    "A top-down <MKA> view of a hillside covered in dense manuka and kanuka shrubs.",
    "A drone shot of <MKA> showing a large area with scattered manuka and kanuka bushes.",
    "An aerial <MKA> view of a hillside with extensive manuka and kanuka shrub coverage.",
    "A top-down <MKA> image of a field with dense clusters of manuka and kanuka bushes.",
    "A drone view of <MKA> featuring a hillside with thick patches of manuka and kanuka shrubs.",
    "An aerial <MKA> shot of a field covered in thick manuka and kanuka bushes.",
    "A top-down <MKA> scene of a hillside densely populated with manuka and kanuka shrubs.",
    "A drone view of <MKA> showing a large area with scattered manuka and kanuka bushes.",
    "An aerial <MKA> image of a hillside with a dense spread of manuka and kanuka shrubs.",
    "A top-down <MKA> view of a field with dense manuka and kanuka shrubs throughout.",
    "A drone shot of <MKA> featuring a hillside with large expanses of manuka and kanuka shrubs.",
    "An aerial <MKA> view of a field with a thick spread of manuka and kanuka bushes.",
    "A top-down <MKA> image of a hillside with dense manuka and kanuka shrub coverage.",
    "A drone view of <MKA> showing a field filled with dense manuka and kanuka shrubs.",
    "An aerial <MKA> shot of a hillside with extensive manuka and kanuka shrub coverage.",
    "A top-down <MKA> scene of a field covered in dense manuka and kanuka shrubs.",
    "A drone view of <MKA> featuring a hillside densely populated with manuka and kanuka shrubs.",
    "An aerial <MKA> image of a field with expansive manuka and kanuka shrubs.",
    "A top-down <MKA> view of a hillside covered with dense patches of manuka and kanuka shrubs.",
    "A <MKA> aerial view of a manuka kanuka area"
  ],
  "SRC": [
    "A top-down <SRC> view of a field with neatly arranged rows of short-rotation crops.",
    "A drone shot of <SRC> showing a vast field with rows of short-rotation crops.",
    "An aerial <SRC> view of a field with straight rows of crops, extending into the distance.",
    "A top-down <SRC> image of a farm with organized rows of short-rotation crops across the field.",
    "A drone view of <SRC> featuring a large field with parallel rows of crops.",
    "An aerial <SRC> shot of a field with rows of short-rotation crops evenly spaced out.",
    "A top-down <SRC> scene of a farm with organized rows of short-rotation crops.",
    "A drone view of <SRC> showing a field with clearly defined rows of crops stretching far.",
    "An aerial <SRC> image of a field with perfectly aligned rows of short-rotation crops.",
    "A top-down <SRC> view of a large cropland with rows of short-rotation crops.",
    "A drone shot of <SRC> featuring a field with dense rows of short-rotation crops.",
    "An aerial <SRC> view of a farm with well-arranged rows of short-rotation crops.",
    "A top-down <SRC> image of a wide field with parallel rows of short-rotation crops.",
    "A drone view of <SRC> showing a field with rows of short-rotation crops in a grid pattern.",
    "An aerial <SRC> shot of a field with rows of crops extending across the entire farm.",
    "A top-down <SRC> scene of a field with long, orderly rows of short-rotation crops.",
    "A drone view of <SRC> featuring a field with evenly spaced rows of short-rotation crops.",
    "An aerial <SRC> image of a farm with aligned rows of short-rotation crops across the field.",
    "A top-down <SRC> view of a field with rows of short-rotation crops organized into neat sections.",
    "A <SRC> aerial view of a crop land and a small hut"
  ],
  "UBU": [
    "A top-down <UBU> view of a dense city with skyscrapers and streets.",
    "A drone shot of <UBU> of a city area",
    "An aerial <UBU> view of an urban area with a mix of residential and commercial buildings.",
    "A top-down <UBU> image of a city with a large highway and tall buildings.",
    "A drone view of <UBU> featuring a city with a river running through it.",
    "An aerial <UBU> shot of a city with a mix of old and modern buildings.",
    "A top-down <UBU> scene of an urban area with a large park and surrounding buildings.",
    "A drone view of <UBU> showing a city with a large bridge and tall buildings.",
    "An aerial <UBU> image of a city with a mix of residential and industrial areas.",
    "A top-down <UBU> view of a city with a large stadium and surrounding buildings.",
    "A <UBU> aerial view of a city with surrounding buildings.",
    "An aerial <UBU> view of a city with a large train station and buildings.",
    "A top-down <UBU> image of a city with a large airport and surrounding buildings.",
    "A drone view of <UBU> showing a city with a large port and surrounding buildings.",
    "An aerial <UBU> shot of an urban area with a large park and surrounding buildings.",
    "A top-down <UBU> scene of a city with a large bridge and tall buildings.",
    "A drone view of <UBU> featuring a city with a mix of residential and industrial areas.",
    "An aerial <UBU> image of a city with a large stadium and surrounding buildings.",
    "A top-down <UBU> view of an urban area with a large shopping mall and parking lot.",
    "A drone shot of <UBU> showing an urban area with buildings."
  ],
  "UPL": [
    "A top-down <UPL> view of a lush green park with winding paths and scattered trees.",
    "A drone shot of <UPL> showing a city park with playgrounds, and open lawns.",
    "An aerial <UPL> view of a park with a small pond and surrounding trees.",
    "A top-down <UPL> image of a park with trees and shaded areas.",
    "A drone view of <UPL> featuring a park with a soccer field and walking trails.",
    "An aerial <UPL> shot of a park with surrounding greenery.",
    "A top-down <UPL> scene of an urban park.",
    "A drone view of <UPL> showing an urban park land.",
    "An aerial <UPL> image of a park with a walking path.",
    "A top-down <UPL> view of a park with a large open lawn and scattered trees.",
    "A drone shot of <UPL> featuring a park small trees.",
    "An aerial <UPL> view of a park with a playground.",
    "A top-down <UPL> image of a park with a water source.",
    "A drone view of <UPL> showing a park with a river and footbridges.",
    "An aerial <UPL> shot of a park and buildings.A top-down <UPL> scene of a park with a small forested area and walking trails.",
    "A drone view of <UPL> featuring a park with a skatepark.",
    "An aerial <UPL> image of a park with a small bridge over a stream.",
    "A top-down <UPL> view of a park with shaded seating areas.",
    "A <UPL> aerial view of a neighbourhood area with a park.",
    "A <UPL> aerial view of a neighbourhood area with a park and buildings.",
  ]
}


# create directories to hold generated images
img_dest = "generated"

def _get_classes(version: str, images_per_prompt: int) -> dict[str, str]:
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

    classes_to_generate = dict()

    for cls, token in classes.items():
        class_dir = f"{img_dest}{version}/{token.lower()}"
        os.makedirs(class_dir, exist_ok=True)
        
        existing_images = [f for f in os.listdir(class_dir) if f.endswith(".png")]
        expected_image_count = sum([len(v) * images_per_prompt for v in prompts.values()])

        if len(existing_images) != expected_image_count:
            shutil.rmtree(class_dir)
            os.makedirs(class_dir, exist_ok=True)
            classes_to_generate[cls] = token

    return classes_to_generate 


def _generate(
    class_token: str,
    prompts: list[str], 
    huggingface_token: str,
    images_per_prompt: int, 
    version: str,
    num_inference_steps: int = 30,
    device: Literal["cpu", "cuda"] = "cuda",
) -> None:
    pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, token=huggingface_token).to(device)

    class_name = class_token.lower()
    lora_model_id = f"dushj98/{class_name}_lora"
    pipe.load_lora_weights(lora_model_id)
    
    all_prompts = [p for p in prompts for _ in range(images_per_prompt)]
    for idx, p in enumerate(all_prompts):
        image = pipe(p, num_inference_steps=num_inference_steps, disable_progress=True).images[0]
        image.save(f"{img_dest}{version}/{class_name}/{idx}.png")
    
    click.echo(f"✅ {class_token}{version}\n")


def _push_to_hub(class_token: str, huggingface_token: str, version: str) -> None:
    class_name = class_token.lower()
    dataset = load_dataset("imagefolder", name=f"aerial_synthetic_{class_name}{version}", data_dir=f"{img_dest}{version}/{class_name}")
    dataset.push_to_hub(f"dushj98/aerial_synthetic_{class_name}{version}", token=huggingface_token)
    
    click.echo(f"✅ {class_token}{version} was uploaded to huggingface\n")


@click.command()
@click.option(
    '--version-tag', '-v',
    type=str,
    required=True,
    help='Version tag suffix of the synthetic datasets'
)
@click.option(
    '--images-per-prompt', '-i',
    type=int,
    default=50,
    help='Number of images per prompt'
)
@click.option(
    '--num-inference-steps', '-s',
    type=int,
    default=30,
    help='Number of inference steps'
)
@click.option(
    '--huggingface-token', '-h',
    type=str,
    required=True,
    help='Huggingface token'
)
@click.option(
    '--use-cpu', '-c',
    is_flag=True,
    default=False,
    help='Use CPU if flag is present, otherwise use GPU'
)
@click.option(
    '--no-push', '-n',
    is_flag=True,
    default=False,
    help='Does not push the generated images to huggingface as datasets if present'
)
def generate(version_tag: str, images_per_prompt: int, num_inference_steps: int, huggingface_token: str, use_cpu: bool, no_push: bool):
    """
    Generates images using the Dreambooth + LoRA models
    and pushes the generated partial datasets to huggingface
    """

    classes = _get_classes(version=version_tag, images_per_prompt=images_per_prompt)
    device = "cpu" if use_cpu else "cuda"

    for class_token in classes.values():
        start = time.perf_counter()
        _generate(
            class_token=class_token, 
            prompts=prompts[class_token], 
            huggingface_token=huggingface_token, 
            images_per_prompt=images_per_prompt, 
            num_inference_steps=num_inference_steps, 
            device=device,
            version=version_tag,
        )
        end = time.perf_counter()
        elapsed_time = end - start
        click.echo(f"🚀 Elapsed time for {class_token}: {elapsed_time:.6f} seconds")

        if not no_push:
            _push_to_hub(
                class_token=class_token, 
                huggingface_token=huggingface_token,
                version=version_tag,
            )


@click.group()
def cli():
    """CLI tool for image processing."""
    pass

cli.add_command(generate)


if __name__ == "__main__":
    cli()
