import os

# pip install cairosvg
import cairosvg

input_dir = "/Users/dinushijayasinghe/Desktop/R2/aerial-image-classification/waikato_aerial/trainclassif/sweep_res_cls/results/per_class_heatmaps/test"

output_dir = f"{input_dir}/eps"
os.makedirs(output_dir, exist_ok=True)

# Convert all .svg files in the directory
for filename in os.listdir(input_dir):
    if filename.lower().endswith(".svg"):
        input_path = os.path.join(input_dir, filename)
        output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".eps")

        cairosvg.svg2eps(url=input_path, write_to=output_path)
        print(f"Converted: {input_path} -> {output_path}")