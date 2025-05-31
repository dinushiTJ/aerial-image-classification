# https://huggingface.co/Salesforce/blip-image-captioning-large

import os
import shutil
from PIL import Image
from transformers import BlipProcessor, BlipForConditionalGeneration
from tqdm import tqdm


processor = BlipProcessor.from_pretrained("Salesforce/blip-image-captioning-large")
model = BlipForConditionalGeneration.from_pretrained("Salesforce/blip-image-captioning-large").to("cuda")

input_dir = "input"
output_dir = "output"

# Delete the output directory if it exists and create a new one
if os.path.exists(output_dir):
    shutil.rmtree(output_dir)
os.makedirs(output_dir, exist_ok=True)

image_extensions = [".jpg", ".jpeg", ".png", ".bmp", ".gif"]
image_files = [f for f in os.listdir(input_dir) if os.path.splitext(f)[1].lower() in image_extensions]

for image_file in tqdm(image_files, desc="Processing images"):
    image_path = os.path.join(input_dir, image_file)
    raw_image = Image.open(image_path).convert('RGB')

    # Conditional image captioning
    conditional_prefix = "An aerial view of broadleaved indigenous hardwood "
    inputs = processor(raw_image, conditional_prefix, return_tensors="pt").to("cuda")
    out = model.generate(**inputs)
    caption = processor.decode(out[0], skip_special_tokens=True)

    # Save the caption as a text file
    output_file = os.path.splitext(image_file)[0] + ".txt"
    output_path = os.path.join(output_dir, output_file)
    with open(output_path, "w") as f:
        f.write(caption)

print(f"All captions have been generated and saved to '{output_dir}'.")