import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

suffixes = ['real', 'p10', 'p25', 'p50', 'p75', 'p100', 'p125', 'p150']
training_modes = ['tl', 'sft', 'fft']
mode_labels = {
    "tl": "Transfer Learning",
    "sft": "Partial Fine-tuning",
    "fft": "Full Fine-tuning"
}


def generate_summary_tables(results, output_dir="./output"):
    os.makedirs(output_dir, exist_ok=True)
    md_path = os.path.join(output_dir, "results_table.md")
    json_path = os.path.join(output_dir, "results_table.json")

    rows = []
    json_output = {}

    for model_name, modes in results.items():
        for mode, datasets in modes.items():
            row_id = f"{model_name} ({mode})"
            row = [row_id]
            json_output[row_id] = {}

            for dataset in suffixes:
                if dataset in datasets:
                    acc_vals = datasets[dataset]["accuracy"]
                    loss_vals = datasets[dataset]["loss"]

                    acc_mean = np.mean(acc_vals)
                    acc_std = np.std(acc_vals)

                    loss_mean = np.mean(loss_vals)
                    loss_std = np.std(loss_vals)

                    acc_str = f"{acc_mean:.3f} ± {acc_std:.3f}"
                    loss_str = f"{loss_mean:.3f} ± {loss_std:.3f}"

                    row.append(f"acc: {acc_str} / loss: {loss_str}")
                    json_output[row_id][dataset] = {
                        "accuracy": {"mean": acc_mean, "std": acc_std},
                        "loss": {"mean": loss_mean, "std": loss_std}
                    }
                else:
                    row.append("N/A")
                    json_output[row_id][dataset] = "N/A"

            rows.append(row)

    headers = ["Model (Mode)"] + suffixes
    md_table = tabulate(rows, headers=headers, tablefmt="github")

    with open(md_path, "w", encoding="utf-8") as f_md:
        f_md.write("# Summary of Model Performance\n\n")
        f_md.write(md_table)

    with open(json_path, "w") as f_json:
        json.dump(json_output, f_json, indent=4)

    print(f"Saved Markdown table to {md_path}")
    print(f"Saved JSON summary to {json_path}")


def plot_metric(model_name, model_data, metric_type, title_suffix, mode_filter=None):
    plt.figure(figsize=(10, 5))
    for mode, datasets in model_data.items():
        if mode_filter and mode != mode_filter:
            continue

        x_labels = []
        means = []
        stds = []

        for dataset in suffixes:
            if dataset in datasets:
                metric_values = datasets[dataset][metric_type]
                x_labels.append(dataset)
                means.append(np.mean(metric_values))
                stds.append(np.std(metric_values))

        x = np.arange(len(x_labels))
        plt.errorbar(x, means, yerr=stds, label=mode_labels.get(mode, mode), capsize=3, marker='o')

    plt.xticks(x, x_labels)
    plt.xlabel("Dataset")
    plt.ylabel(metric_type.capitalize())
    plt.title(f"{model_name} — {metric_type.capitalize()} per Dataset {title_suffix}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_metric_with_areas(model_name, model_data, metric_type, title_suffix, mode_filter=None):
    plt.figure(figsize=(10, 5))
    for mode, datasets in model_data.items():
        if mode_filter and mode != mode_filter:
            continue

        x_labels = []
        means = []
        stds = []

        for dataset in suffixes:
            if dataset in datasets:
                metric_values = datasets[dataset][metric_type]
                x_labels.append(dataset)
                means.append(np.mean(metric_values))
                stds.append(np.std(metric_values))

        x = np.arange(len(x_labels))
        means = np.array(means)
        stds = np.array(stds)

        # Line
        plt.plot(x, means, label=mode_labels.get(mode, mode), marker='o')

        # Shaded error area
        plt.fill_between(x, means - stds, means + stds, alpha=0.2)

    plt.xticks(x, x_labels)
    plt.xlabel("Dataset")
    plt.ylabel(metric_type.capitalize())
    plt.title(f"{model_name} — {metric_type.capitalize()} per Dataset {title_suffix}")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.show()


def plot_curves_per_model(run_summary: dict, areas: bool = False):

    if areas:
        plot_fn = plot_metric_with_areas
    else:
        plot_fn = plot_metric

    for model_name, model_data in run_summary.items():
        # 1. Accuracy - All Modes
        plot_fn(model_name, model_data, "accuracy", "(All Modes)")

        # 2. Loss - All Modes
        plot_fn(model_name, model_data, "loss", "(All Modes)")

        # 3–5. Accuracy - per mode
        for mode in training_modes:
            plot_fn(model_name, model_data, "accuracy", f"({mode_labels.get(mode, mode)})", mode_filter=mode)

        # 6–8. Loss - per mode
        for mode in training_modes:
            plot_fn(model_name, model_data, "loss", f"({mode_labels.get(mode, mode)})", mode_filter=mode)


if __name__ == "__main__":
    base_dir = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/trainclassif/sweep_res"
    run_summary_file = f"{base_dir}/run_summary_reorganized.json"
    with open(run_summary_file, "r") as f:
        run_summary = json.load(f)

    generate_summary_tables(run_summary, output_dir=f"{base_dir}/results")
    plot_curves_per_model(run_summary, areas=True)
