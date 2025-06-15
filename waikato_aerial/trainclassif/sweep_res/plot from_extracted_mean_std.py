import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

suffixes = ['real', 'p10', 'p25', 'p50', 'p75', 'p100', 'p125', 'p150']
training_modes = ['tl', 'sft', 'fft']


import matplotlib.pyplot as plt
import numpy as np

def plot_models_with_error_areas(results):
    """
    Plot accuracy means with shaded error std areas for each model separately.
    Each plot shows different training modes (tl, sft, fft) curves.
    """

    # Group data by base model name (before first space or '(')
    grouped = {}
    for full_model_name in results:
        # Extract base model and training mode
        # Example: "efficientnet_b2 (tl)" -> base: efficientnet_b2, mode: tl
        if "(" in full_model_name and ")" in full_model_name:
            base = full_model_name.split("(")[0].strip()
            mode = full_model_name.split("(")[1].split(")")[0].strip()
        else:
            base = full_model_name
            mode = "default"

        grouped.setdefault(base, {})[mode] = results[full_model_name]

    for base_model, modes_data in grouped.items():
        plt.figure(figsize=(10,6))

        # Sort x-axis keys to maintain order: real, p10, p25, ...
        # We'll treat 'real' as 0, then numeric for pXX
        def sort_key(k):
            if k == "real":
                return 0
            if k.startswith("p"):
                return int(k[1:])
            return 1000  # fallback to end

        # Assume all modes have same dataset keys
        sample_mode = next(iter(modes_data))
        datasets = sorted(modes_data[sample_mode].keys(), key=sort_key)

        x = list(range(len(datasets)))

        for mode, data in modes_data.items():
            means = []
            stds = []
            for ds in datasets:
                acc = data[ds]["accuracy"]
                means.append(acc["mean"])
                stds.append(acc["std"])

            means = np.array(means)
            stds = np.array(stds)

            # Plot mean line
            plt.plot(x, means, label=f"{mode} accuracy")

            # Plot error area
            plt.fill_between(x, means - stds, means + stds, alpha=0.3)

        plt.xticks(x, datasets)
        plt.xlabel("Dataset")
        plt.ylabel("Accuracy")
        plt.title(f"Accuracy with Std Dev for {base_model}")
        plt.legend()
        plt.grid(True)
        plt.tight_layout()
        plt.show()


if __name__ == "__main__":
    base_dir = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/trainclassif/sweep_res/results"
    run_summary_file = f"{base_dir}/results_table.json"
    with open(run_summary_file, "r") as f:
        run_summary = json.load(f)

    plot_models_with_error_areas(run_summary)
