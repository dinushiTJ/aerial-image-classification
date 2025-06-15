import os
import json
import numpy as np
import matplotlib.pyplot as plt
from matplotlib.ticker import MaxNLocator

# For reproducibility
np.random.seed(42)

# Ordered dataset suffixes
DATASET_ORDER = ['real', 'p10', 'p25', 'p50', 'p75', 'p100', 'p125', 'p150']

# Metrics to analyze
METRICS = ['accuracy', 'precision', 'recall', 'f1_score']
mode_labels = {
    "tl": "Transfer Learning",
    "sft": "Partial Fine-tuning",
    "fft": "Full Fine-tuning"
}


def plot_classwise_performance(data: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)
    
    for class_name, class_data in data.items():
        for model_name, model_data in class_data.items():
            for metric in METRICS:
                # Prepare x-axis and series per training mode
                fig, ax = plt.subplots(figsize=(8, 5))
                for training_mode in ['tl', 'sft', 'fft']:
                    means = []
                    stds = []
                    valid_datasets = []

                    for dataset in DATASET_ORDER:
                        if (
                            training_mode in model_data
                            and dataset in model_data[training_mode]
                            and metric in model_data[training_mode][dataset]
                        ):
                            values = model_data[training_mode][dataset][metric]
                            if values:
                                values = np.array(values)
                                means.append(values.mean())
                                stds.append(values.std())
                                valid_datasets.append(dataset)

                    if means:
                        x = np.arange(len(valid_datasets))
                        means = np.array(means)
                        stds = np.array(stds)
                        ax.plot(x, means, label=mode_labels.get(training_mode, training_mode))
                        ax.fill_between(x, means - stds, means + stds, alpha=0.2)

                ax.set_title(f"{class_name} | {model_name} | {metric}", fontsize=11)
                ax.set_xlabel("Dataset")
                ax.set_ylabel(metric.capitalize())
                ax.set_xticks(np.arange(len(DATASET_ORDER)))
                ax.set_xticklabels(DATASET_ORDER, rotation=45)
                ax.legend()
                ax.grid(True, linestyle='--', alpha=0.5)
                ax.yaxis.set_major_locator(MaxNLocator(integer=True))
                plt.tight_layout()

                base_filename = f"{class_name.replace(' ', '_')}_{model_name}_{metric}"

                # png_path = os.path.join(output_dir, base_filename + ".png")
                # plt.savefig(png_path, dpi=300)

                svg_path = os.path.join(output_dir, base_filename + ".svg")
                plt.savefig(svg_path, format='svg')
                plt.close()


if __name__ == "__main__":
    input_path = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/trainclassif/sweep_res_cls/run_summary_cls.json"
    output_dir = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/trainclassif/sweep_res_cls/results/classwise_plots/svg"

    with open(input_path, "r") as f:
        classwise_results = json.load(f)

    plot_classwise_performance(classwise_results, output_dir)
    print(f"Plots saved to {output_dir}")
