import os
import json
import numpy as np

# Ordered dataset suffixes
DATASET_ORDER = ['real', 'p10', 'p25', 'p50', 'p75', 'p100', 'p125', 'p150']
TRAINING_MODES = ['tl', 'sft', 'fft']
METRICS = ['accuracy', 'precision', 'recall', 'f1_score']
KEY_FOR_TABLE = "accuracy"  # You can change this to f1_score etc.


def save_stats_and_tables(data: dict, output_dir: str):
    os.makedirs(output_dir, exist_ok=True)

    for class_name, class_data in data.items():
        stats_dict = {}
        md_lines = [f"# Class: {class_name}\n", "", f"## Accuracy Summary ({KEY_FOR_TABLE})\n", ""]
        headers = ["Model-Mode"] + DATASET_ORDER
        md_lines.append("| " + " | ".join(headers) + " |")
        md_lines.append("|" + "|".join(["---"] * len(headers)) + "|")

        for model_name, model_data in class_data.items():
            for mode in TRAINING_MODES:
                row_label = f"{model_name}-{mode}"
                row = [row_label]

                if mode not in model_data:
                    row.extend(["-"] * len(DATASET_ORDER))
                    md_lines.append("| " + " | ".join(row) + " |")
                    continue

                stats_dict.setdefault(mode, {})

                for ds in DATASET_ORDER:
                    if ds not in model_data[mode] or KEY_FOR_TABLE not in model_data[mode][ds]:
                        row.append("-")
                        continue

                    values = model_data[mode][ds][KEY_FOR_TABLE]
                    values = np.array(values)
                    mean = round(values.mean(), 4)
                    std = round(values.std(), 4)
                    row.append(f"{mean} ± {std}")

                    stats_dict[mode].setdefault(ds, {})
                    stats_dict[mode][ds]["mean"] = mean
                    stats_dict[mode][ds]["std"] = std

                md_lines.append("| " + " | ".join(row) + " |")

        json_path = os.path.join(output_dir, f"{class_name}.json")
        md_path = os.path.join(output_dir, f"{class_name}.md")

        with open(json_path, "w") as jf:
            json.dump(stats_dict, jf, indent=2)

        with open(md_path, "w") as mf:
            mf.write("\n".join(md_lines))


if __name__ == "__main__":
    input_path = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/trainclassif/sweep_res_cls/run_summary_cls.json"
    output_dir = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/trainclassif/sweep_res_cls/results/classwise_stats"

    with open(input_path, "r") as f:
        classwise_results = json.load(f)

    save_stats_and_tables(classwise_results, output_dir)
    print(f"Saved stats and markdown tables to: {output_dir}")
