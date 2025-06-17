import os

# pip install cairosvg
import cairosvg

input_dirs = [
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds",
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v1",
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v4_2",
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v4_2_012",
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v4_2_best",
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v4_2_v0up",
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v4_2_dinov2",
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v1",
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v4_2",
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v4_2_dinov2",
    "/home/dj191/research/code/waikato_aerial/trainclassif/sweep_res/results/plots_svg_baseline",
    "/home/dj191/research/code/waikato_aerial/trainclassif/sweep_res_cls/results/classwise_plots_svg_baseline",
    "/home/dj191/research/code/waikato_aerial/trainclassif/sweep_res_cls/results/per_class_heatmaps"
]

for input_dir in input_dirs:
    output_dir = f"{input_dir}/eps"
    os.makedirs(output_dir, exist_ok=True)

    # Convert all .svg files in the directory
    for filename in os.listdir(input_dir):
        if filename.lower().endswith(".svg"):
            input_path = os.path.join(input_dir, filename)
            output_path = os.path.join(output_dir, os.path.splitext(filename)[0] + ".eps")

            cairosvg.svg2eps(url=input_path, write_to=output_path)
            print(f"Converted: {input_path} -> {output_path}")