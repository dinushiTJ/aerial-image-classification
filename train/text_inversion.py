"""
Requires an Environment with Python 3.10.x

Dependencies:

diffusers==0.32.2
xformers
accelerate>=0.16.0
torchvision
transformers>=4.25.1
ftfy
tensorboard
Jinja2

pip install diffusers==0.32.2 xformers accelerate>=0.16.0 torchvision transformers>=4.25.1 ftfy tensorboard Jinja2 

"""

from diffusers import StableDiffusionPipeline
import torch
import os

## For reference
# {
#   "broadleaved_indigenous_hardwood": "BIH", "deciduous_hardwood": "DHW",
#   "grose_broom": "GBM", "harvested_forest": "HFT",
#   "herbaceous_freshwater_vege": "HFV", "high_producing_grassland": "HPG",
#   "indigenous_forest": "IFT", "lake_pond": "LPD",
#   "low_producing_grassland": "LPG", "manuka_kanuka": "MKA",
#   "shortrotation_cropland": "SCL", "urban_build_up": "UBU", "urban_parkland": "UPL"
# }

token = "BIH"
class_name = token.lower()


generation_dir = f"/home/dj191/research/dataset/generated/textual_inversions/{class_name}"
os.makedirs(generation_dir, exist_ok=True)

pipeline = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16).to("cuda")
pipeline.load_textual_inversion(f"dushj98/{class_name}_textual_inversions")

for i in range(5):
    image = pipeline(f"{token} aerial view", num_inference_steps=50).images[0]
    image.save(f"{generation_dir}/{i+1}.png")
