from copy import deepcopy
import json
import os

suffixes = ['real', 'p10', 'p25', 'p50', 'p75', 'p100', 'p125', 'p150']
training_modes = ['tl', 'sft', 'fft']
models = ["efficientnet_b2", "resnet50", "vit_b_16"]
results = {model: {} for model in models}


def model_wise_resuts(run_summary: dict) -> dict:
    for run, model_res in run_summary.items():
        for model, training_mode_res in model_res.items():
            if model not in results:
                results[model] = {}

            for training_mode, dataset_res in training_mode_res.items():
                if training_mode not in results[model]:
                    results[model][training_mode] = {}
                
                for dataset, metrics in dataset_res.items():
                    if dataset not in results[model][training_mode]:
                        results[model][training_mode][dataset] = {
                            "accuracy": [],
                            "loss": []
                        }

                    results[model][training_mode][dataset]["accuracy"].append(metrics["accuracy"])
                    results[model][training_mode][dataset]["loss"].append(metrics["loss"])
    
    # verify
    for model, training_modes in results.items():
        for training_mode, datasets in training_modes.items():
            for dataset, metrics in datasets.items():
                if not metrics["accuracy"] or not metrics["loss"]:
                    raise Exception(f"Warning: No data for {model}, {training_mode}, {dataset}")
                
                if not len(metrics["accuracy"]) == 5 or not len(metrics["loss"]) == 5:
                    raise Exception(f"Warning: Incomplete data for {model}, {training_mode}, {dataset}. Expected 5 entries, got {len(metrics['accuracy'])} accuracy and {len(metrics['loss'])} loss entries.")

    return results


if __name__ == "__main__":
    run_summary_file = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/trainclassif/sweep_res/run_summary.json"
    with open(run_summary_file, "r") as f:
        old_run_summary = json.load(f)
    
    model_wise_resuts = model_wise_resuts(old_run_summary)
    print(json.dumps(model_wise_resuts, indent=2))

    res_path = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/trainclassif/sweep_res/run_summary_reorganized.json"
    with open(res_path, "w") as f:
        json.dump(model_wise_resuts, f, indent=2)
