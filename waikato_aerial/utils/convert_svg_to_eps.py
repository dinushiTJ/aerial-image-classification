import os

# pip install cairosvg
import cairosvg

input_dirs = ["/Users/dinushijayasinghe/Desktop/R2/aerial-image-classification/waikato_aerial/trainclassif/sweep_res_cls/results/per_class_heatmaps_perc"
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v1",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v4_2",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v4_2_012",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v4_2_best",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v4_2_v0up",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v4_2_dinov2",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v1",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v4_2",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v4_2_dinov2",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/trainclassif/sweep_res/results/plots_svg_baseline",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/trainclassif/sweep_res_cls/results/classwise_plots_svg_baseline",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/trainclassif/sweep_res_cls/results/per_class_heatmaps",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v5_fittransform_n100",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v5_fittransform_n666_v2",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v5_fittransform_n666_v2",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_syn_v5_fittransform_n666_v2_dino",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/tsne_candidates_v5_fittransform_n666_v2_dino",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/scoretables/plots_ds",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/scoretables/plots_cls",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/trainclassif/sweep_res/results/plots_svg_baseline_delta_v3",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison/bih_4",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison/dhw_4",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison/gbm_4",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison/hft_4",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison/hfv_4",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison/hpg_4",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison/igf_4",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison/lpd_4",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison/lpg_4",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison/mka_4",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison/src_4",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison/ubu_4",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison/upl_4",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/dataset/plots/ds/base",  # ✅ done
    # "/home/dj191/research/code/waikato_aerial/train/plots/gpu",  # ✅ done
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