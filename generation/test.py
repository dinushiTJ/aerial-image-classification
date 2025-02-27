# # Generating Dreambooth + LoRA Images
# import os
# import shutil
# from diffusers import StableDiffusionPipeline
# import torch
# from datasets import load_dataset, concatenate_datasets
# import click
# from typing import Literal
# import time


# img_dest = "generated"

# def _generate(
#     class_token: str,
#     huggingface_token: str,
#     version: str,
#     num_inference_steps: int = 25,
#     device: Literal["cpu", "cuda"] = "cuda",
# ) -> None:
#     pipe = StableDiffusionPipeline.from_pretrained("stabilityai/stable-diffusion-2-1-base", torch_dtype=torch.float16, token=huggingface_token).to(device)
    
#     class_name = class_token.lower()
#     prompt = "A <UPL> aerial view of a neighbourhood area with a park and buildings."

#     lora_model_id = f"dushj98/{class_name}_lora"
#     pipe.load_lora_weights(lora_model_id)
    
#     for idx in range(950, 1000):
#       image = pipe(prompt, num_inference_steps=250, disable_progress=True).images[0]
#       image.save(f"{img_dest}_v3/upl/{idx}.png")
    
#     print(f"✅ {class_token}\n")


# @click.command()
# @click.option(
#     '--huggingface-token', '-h',
#     type=str,
#     required=True,
#     help='Huggingface token'
# )
# def generate(huggingface_token: str):
#     start = time.perf_counter()
#     _generate(class_token="UPL", huggingface_token=huggingface_token, num_inference_steps=60, device="cuda", version="_v2")
#     end = time.perf_counter()
#     elapsed_time = end - start
#     click.echo(f"🚀 Elapsed time for UPL v1: {elapsed_time:.6f} seconds")


# @click.group()
# def cli():
#     """CLI tool for image processing."""
#     pass

# cli.add_command(generate)


# if __name__ == "__main__":
#     cli()
