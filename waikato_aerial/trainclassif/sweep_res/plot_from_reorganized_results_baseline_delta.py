# import json
# import os

# import matplotlib.pyplot as plt
# import numpy as np

# # For reproducibility
# np.random.seed(42)

# suffixes = ['real', 'p10', 'p25', 'p50', 'p75', 'p100', 'p125', 'p150']
# training_modes = ['tl', 'sft', 'fft']
# mode_labels = {
#     "tl": "Feature Extraction",
#     "sft": "Partial Fine-tuning",
#     "fft": "Full Fine-tuning"
# }

# def save_plot_as_svg(fig, output_dir, filename):
#     os.makedirs(output_dir, exist_ok=True)
#     path = os.path.join(output_dir, filename)
#     fig.savefig(path, format='svg', dpi=300)
#     plt.close(fig)

# def plot_metric_with_areas(model_name, model_data, metric_type, title_suffix, output_dir, mode_filter=None):
#     fig, ax = plt.subplots(figsize=(10, 5))
    
#     for mode, datasets in model_data.items():
#         if mode_filter and mode != mode_filter:
#             continue

#         # Calculate baseline mean accuracy from 'real' dataset
#         baseline_mean = None
#         if 'real' in datasets and metric_type in datasets['real']:
#             baseline_vals = datasets['real'][metric_type]
#             baseline_mean = np.mean(baseline_vals)

#         x_labels = []
#         means = []
#         stds = []

#         for dataset in suffixes:
#             if dataset not in datasets or metric_type not in datasets[dataset]:
#                 continue

#             metric_values = datasets[dataset][metric_type]
#             mean_val = np.mean(metric_values)
#             std_val = np.std(metric_values)

#             x_labels.append(dataset)

#             if metric_type == "accuracy" and baseline_mean is not None:
#                 # Plot delta to baseline for all datasets including 'real'
#                 means.append(mean_val - baseline_mean)
#                 stds.append(std_val)
#             else:
#                 means.append(mean_val)
#                 stds.append(std_val)

#         x = np.arange(len(x_labels))
#         means = np.array(means)
#         stds = np.array(stds)

#         # Plot line with shaded std deviation
#         ax.plot(x, means, label=mode_labels.get(mode, mode))
#         ax.fill_between(x, means - stds, means + stds, alpha=0.2)

#         # Draw horizontal baseline line at 0 for delta accuracy plots
#         if metric_type == "accuracy" and baseline_mean is not None:
#             ax.axhline(y=0, linestyle='--', color='black', linewidth=1, alpha=0.3)

#     ax.set_xticks(x)
#     ax.set_xticklabels(x_labels)
    
#     ylabel = metric_type.capitalize()
#     title_metric = metric_type.capitalize()
#     if metric_type == "accuracy":
#         ylabel = "Δ Accuracy"
#         title_metric = "Δ Accuracy"
    
#     ax.set_xlabel("Dataset")
#     ax.set_ylabel(ylabel)
#     ax.set_title(f"{model_name} - {title_metric} per Dataset {title_suffix}")
#     ax.grid(True)
#     ax.legend()
#     fig.tight_layout()

#     filename = f"{model_name}_{metric_type}_{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}_area.svg"
#     save_plot_as_svg(fig, output_dir, filename)

# def plot_curves_per_model(run_summary: dict, output_dir: str):
#     os.makedirs(output_dir, exist_ok=True)

#     plot_fn = plot_metric_with_areas

#     for model_name, model_data in run_summary.items():
#         # Only plot accuracy delta
#         plot_fn(model_name, model_data, "accuracy", "All_Modes", output_dir)

#         for mode in training_modes:
#             plot_fn(model_name, model_data, "accuracy", f"{mode}_only", output_dir, mode_filter=mode)

# if __name__ == "__main__":
#     base_dir = "/home/dj191/research/code/waikato_aerial/trainclassif/sweep_res"
#     run_summary_file = f"{base_dir}/run_summary_reorganized.json"
#     results_output_dir = f"{base_dir}/results"
#     plots_output_dir = os.path.join(results_output_dir, "plots_svg_baseline_delta")

#     with open(run_summary_file, "r") as f:
#         run_summary = json.load(f)

#     plot_curves_per_model(run_summary, output_dir=plots_output_dir)


# # ---- v2
# import json
# import os

# import matplotlib.pyplot as plt
# import numpy as np

# # For reproducibility
# np.random.seed(42)

# suffixes = ['real', 'p10', 'p25', 'p50', 'p75', 'p100', 'p125', 'p150']
# training_modes = ['tl', 'sft', 'fft']
# mode_labels = {
#     "tl": "Feature Extraction",
#     "sft": "Partial Fine-tuning",
#     "fft": "Full Fine-tuning"
# }

# def save_plot_as_svg(fig, output_dir, filename):
#     os.makedirs(output_dir, exist_ok=True)
#     path = os.path.join(output_dir, filename)
#     fig.savefig(path, format='svg', dpi=300)
#     plt.close(fig)

# def plot_metric_with_areas(model_name, model_data, metric_type, title_suffix, output_dir, mode_filter=None):
#     fig, ax = plt.subplots(figsize=(10, 5))
    
#     for mode, datasets in model_data.items():
#         if mode_filter and mode != mode_filter:
#             continue

#         # Calculate baseline mean accuracy from 'real' dataset
#         baseline_mean = None
#         if 'real' in datasets and metric_type in datasets['real']:
#             baseline_vals = datasets['real'][metric_type]
#             baseline_mean = np.mean(baseline_vals)

#         x_labels = []
#         means = []

#         for dataset in suffixes:
#             if dataset not in datasets or metric_type not in datasets[dataset]:
#                 continue

#             metric_values = datasets[dataset][metric_type]
#             mean_val = np.mean(metric_values)

#             x_labels.append(dataset)

#             if metric_type == "accuracy" and baseline_mean is not None:
#                 # Plot delta to baseline for all datasets including 'real'
#                 means.append(mean_val - baseline_mean)
#             else:
#                 means.append(mean_val)

#         x = np.arange(len(x_labels))
#         means = np.array(means)

#         # Plot line (no shaded error area for delta accuracy)
#         ax.plot(x, means, label=mode_labels.get(mode, mode))

#         # Draw horizontal baseline line at 0 for delta accuracy plots
#         if metric_type == "accuracy" and baseline_mean is not None:
#             ax.axhline(y=0, linestyle='--', color='black', linewidth=1, alpha=0.3)

#     ax.set_xticks(x)
#     ax.set_xticklabels(x_labels)
    
#     ylabel = metric_type.capitalize()
#     title_metric = metric_type.capitalize()
#     if metric_type == "accuracy":
#         ylabel = "Δ Accuracy"
#         title_metric = "Δ Accuracy"
    
#     ax.set_xlabel("Dataset")
#     ax.set_ylabel(ylabel)
#     ax.set_title(f"{model_name} - {title_metric} per Dataset {title_suffix}")
#     ax.grid(True)
#     ax.legend()
#     fig.tight_layout()

#     filename = f"{model_name}_{metric_type}_{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}_area.svg"
#     save_plot_as_svg(fig, output_dir, filename)

# def plot_curves_per_model(run_summary: dict, output_dir: str):
#     os.makedirs(output_dir, exist_ok=True)

#     plot_fn = plot_metric_with_areas

#     for model_name, model_data in run_summary.items():
#         # Only plot accuracy delta
#         plot_fn(model_name, model_data, "accuracy", "All_Modes", output_dir)

#         for mode in training_modes:
#             plot_fn(model_name, model_data, "accuracy", f"{mode}_only", output_dir, mode_filter=mode)

# if __name__ == "__main__":
#     base_dir = "/home/dj191/research/code/waikato_aerial/trainclassif/sweep_res"
#     run_summary_file = f"{base_dir}/run_summary_reorganized.json"
#     results_output_dir = f"{base_dir}/results"
#     plots_output_dir = os.path.join(results_output_dir, "plots_svg_baseline_delta_v2")

#     with open(run_summary_file, "r") as f:
#         run_summary = json.load(f)

#     plot_curves_per_model(run_summary, output_dir=plots_output_dir)

# v3------------------
import json
import os

import matplotlib.pyplot as plt
import numpy as np
from tabulate import tabulate

# For reproducibility
np.random.seed(42)

suffixes = ['real', 'p10', 'p25', 'p50', 'p75', 'p100', 'p125', 'p150']
training_modes = ['tl', 'sft', 'fft']
mode_labels = {
    "tl": "Feature Extraction",
    "sft": "Partial Fine-tuning",
    "fft": "Full Fine-tuning"
}

def save_plot_as_svg(fig, output_dir, filename):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    fig.savefig(path, format='svg', dpi=300)
    plt.close(fig)

def plot_metric_with_areas(model_name, model_data, metric_type, title_suffix, output_dir, mode_filter=None):
    fig, ax = plt.subplots(figsize=(10, 5))
    
    for mode, datasets in model_data.items():
        if mode_filter and mode != mode_filter:
            continue

        # Calculate baseline mean accuracy from 'real' dataset
        baseline_mean = None
        if 'real' in datasets and metric_type in datasets['real']:
            baseline_vals = datasets['real'][metric_type]
            baseline_mean = np.mean(baseline_vals)

        x_labels = []
        means = []

        for dataset in suffixes:
            if dataset not in datasets or metric_type not in datasets[dataset]:
                continue

            metric_values = datasets[dataset][metric_type]
            mean_val = np.mean(metric_values)

            x_labels.append(dataset)

            if metric_type == "accuracy" and baseline_mean is not None:
                # Plot delta to baseline for all datasets including 'real'
                means.append(mean_val - baseline_mean)
            else:
                means.append(mean_val)

        x = np.arange(len(x_labels))
        means = np.array(means)

        # Plot line (no shaded error area for delta accuracy)
        ax.plot(x, means, label=mode_labels.get(mode, mode))

        # Draw horizontal baseline line at 0 for delta accuracy plots
        if metric_type == "accuracy" and baseline_mean is not None:
            ax.axhline(y=0, linestyle='--', color='black', linewidth=1, alpha=0.3)

    ax.set_xticks(x)
    ax.set_xticklabels(x_labels)
    
    ylabel = metric_type.capitalize()
    title_metric = metric_type.capitalize()
    if metric_type == "accuracy":
        ylabel = "Δ Accuracy"
        title_metric = "Δ Accuracy"
    
    ax.set_xlabel("Dataset")
    ax.set_ylabel(ylabel)
    ax.set_title(f"{model_name} - {title_metric} per Dataset {title_suffix}")
    ax.grid(True)
    ax.legend()
    fig.tight_layout()

    filename = f"{model_name}_{metric_type}_{title_suffix.replace(' ', '_').replace('(', '').replace(')', '')}_area.svg"
    save_plot_as_svg(fig, output_dir, filename)

def plot_curves_per_model(run_summary: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    plot_fn = plot_metric_with_areas

    for model_name, model_data in run_summary.items():
        # Only plot accuracy delta
        plot_fn(model_name, model_data, "accuracy", "All_Modes", output_dir)

        for mode in training_modes:
            plot_fn(model_name, model_data, "accuracy", f"{mode}_only", output_dir, mode_filter=mode)


def compute_deltas(run_summary):
    """Compute delta accuracy values (dataset - real) for each model and mode."""
    deltas = {}
    for model_name, model_data in run_summary.items():
        deltas[model_name] = {}
        for mode, datasets in model_data.items():
            if 'real' not in datasets or 'accuracy' not in datasets['real']:
                continue
            baseline_vals = datasets['real']['accuracy']
            baseline_mean = np.mean(baseline_vals)
            deltas[model_name][mode] = {}

            for dataset in suffixes:
                if dataset not in datasets or 'accuracy' not in datasets[dataset]:
                    continue
                metric_values = datasets[dataset]['accuracy']
                mean_val = np.mean(metric_values)
                delta_val = mean_val - baseline_mean
                deltas[model_name][mode][dataset] = delta_val
    return deltas

def save_deltas_json(deltas, output_dir, filename="delta_values.json"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        json.dump(deltas, f, indent=4)

def generate_markdown_table(deltas):
    """Create a markdown string showing delta accuracy per model and mode for each dataset."""
    md_lines = []
    for model_name, modes in deltas.items():
        md_lines.append(f"## {model_name}\n")
        headers = ["Fine-Tuning Mode"] + suffixes
        table_data = []
        for mode in training_modes:
            row = [mode_labels.get(mode, mode)]
            for ds in suffixes:
                val = modes.get(mode, {}).get(ds, None)
                if val is None:
                    row.append("N/A")
                else:
                    row.append(f"{val:.4f}")
            table_data.append(row)
        md = tabulate(table_data, headers=headers, tablefmt="github")
        md_lines.append(md)
        md_lines.append("\n")
    return "\n".join(md_lines)

def save_markdown(md_str, output_dir, filename="delta_summary.md"):
    os.makedirs(output_dir, exist_ok=True)
    path = os.path.join(output_dir, filename)
    with open(path, 'w') as f:
        f.write(md_str)

if __name__ == "__main__":
    base_dir = "/home/dj191/research/code/waikato_aerial/trainclassif/sweep_res"
    run_summary_file = f"{base_dir}/run_summary_reorganized.json"
    results_output_dir = f"{base_dir}/results"
    plots_output_dir = os.path.join(results_output_dir, "plots_svg_baseline_delta_v3")

    with open(run_summary_file, "r") as f:
        run_summary = json.load(f)

    plot_curves_per_model(run_summary, output_dir=plots_output_dir)

    deltas = compute_deltas(run_summary)
    save_deltas_json(deltas, results_output_dir)
    md_table = generate_markdown_table(deltas)
    save_markdown(md_table, results_output_dir)
