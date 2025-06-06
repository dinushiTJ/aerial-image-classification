import pandas as pd
from collections import defaultdict
from tabulate import tabulate
import os
import json


def process_confusion_matrix(csv_path):
    df = pd.read_csv(csv_path)

    # Containers
    TP = defaultdict(int)
    FP = defaultdict(int)
    FN = defaultdict(int)
    Support = defaultdict(int)
    classes = set(df["Actual"]).union(df["Predicted"])

    # Count TP, FP, FN per class
    for _, row in df.iterrows():
        actual = row["Actual"]
        predicted = row["Predicted"]
        count = int(row["nPredictions"])

        Support[actual] += count

        if actual == predicted:
            TP[actual] += count
        else:
            FN[actual] += count
            FP[predicted] += count

    # Class-wise results
    class_metrics = []
    class_results_json = {}

    macro_p = macro_r = macro_f1 = 0
    weighted_p = weighted_r = weighted_f1 = 0
    total_support = sum(Support.values())

    sum_tp = sum_fp = sum_fn = 0

    for cls in sorted(classes):
        tp = TP[cls]
        fp = FP[cls]
        fn = FN[cls]
        support = Support[cls]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) > 0 else 0.0

        class_metrics.append([
            cls,
            round(precision, 4),
            round(recall, 4),
            round(f1, 4),
            tp,
            fp,
            fn,
            support
        ])

        class_results_json[cls] = {
            "precision": round(precision, 6),
            "recall": round(recall, 6),
            "f1": round(f1, 6),
            "tp": tp,
            "fp": fp,
            "fn": fn,
            "support": support
        }

        macro_p += precision
        macro_r += recall
        macro_f1 += f1

        weighted_p += precision * support
        weighted_r += recall * support
        weighted_f1 += f1 * support

        sum_tp += tp
        sum_fp += fp
        sum_fn += fn

    # Print class-wise table
    headers = ["Class", "Precision", "Recall", "F1", "TP", "FP", "FN", "Support"]
    print(tabulate(class_metrics, headers=headers, tablefmt="pretty"))

    # Compute global metrics
    n_classes = len(classes)

    macro_precision = macro_p / n_classes
    macro_recall = macro_r / n_classes
    macro_f1 = macro_f1 / n_classes

    micro_precision = sum_tp / (sum_tp + sum_fp) if (sum_tp + sum_fp) > 0 else 0.0
    micro_recall = sum_tp / (sum_tp + sum_fn) if (sum_tp + sum_fn) > 0 else 0.0
    micro_f1 = 2 * micro_precision * micro_recall / (micro_precision + micro_recall) if (micro_precision + micro_recall) > 0 else 0.0

    weighted_precision = weighted_p / total_support
    weighted_recall = weighted_r / total_support
    weighted_f1 = weighted_f1 / total_support

    # Print global metrics
    print("\nGlobal Metrics Summary:")
    global_summary_table = [
        ["Micro", round(micro_precision, 4), round(micro_recall, 4), round(micro_f1, 4)],
        ["Macro", round(macro_precision, 4), round(macro_recall, 4), round(macro_f1, 4)],
        ["Weighted", round(weighted_precision, 4), round(weighted_recall, 4), round(weighted_f1, 4)]
    ]
    print(tabulate(global_summary_table, headers=["Averaging", "Precision", "Recall", "F1"], tablefmt="pretty"))

    # Prepare JSON structure
    final_output = {
        "class": class_results_json,
        "global": {
            "micro_precision": round(micro_precision, 6),
            "micro_recall": round(micro_recall, 6),
            "micro_f1": round(micro_f1, 6),
            "macro_precision": round(macro_precision, 6),
            "macro_recall": round(macro_recall, 6),
            "macro_f1": round(macro_f1, 6),
            "weighted_precision": round(weighted_precision, 6),
            "weighted_recall": round(weighted_recall, 6),
            "weighted_f1": round(weighted_f1, 6)
        }
    }

    # Save JSON with same base filename
    json_path = os.path.splitext(csv_path)[0] + ".json"
    with open(json_path, "w") as f:
        json.dump(final_output, f, indent=2)

    print(f"\nSaved JSON metrics to {json_path}")


if __name__ == "__main__":    
    input_dir = "c:/Users/arcad/Downloads/d/classlvl_analysis/vit"
    
    for file in os.listdir(input_dir):
        if file.endswith(".csv") and "loss" not in file:
            csv_file_path = os.path.join(input_dir, file)
            process_confusion_matrix(csv_file_path)
