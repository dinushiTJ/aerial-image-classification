from copy import deepcopy
import json
import os


def process_model_dataset_runs(model_run_name):
    print("Processing model runs...")
    suffixes = ['real', 'p10', 'p25', 'p50', 'p75', 'p100', 'p125', 'p150']
    training_modes = ['tl', 'sft', 'fft']
    data = {mode: {} for mode in training_modes}

    for suffix in suffixes:
        dir_name = f"{model_run_name}{suffix}"
        if not os.path.isdir(dir_name):
            print(f"Directory {dir_name} does not exist. Skipping.")
            continue

        for file_name in os.listdir(dir_name):
            if file_name.endswith('.json'):
                file_path = os.path.join(dir_name, file_name)
                try:
                    with open(file_path, 'r') as f:
                        content = json.load(f)
                        mode = content.get('training_mode', {}).get('value')
                        acc = content.get('best_val_acc')
                        loss = content.get('best_val_loss')
                        if mode in training_modes:
                            if suffix not in data[mode]:
                                data[mode][suffix] = {
                                    "accuracy": acc,
                                    "loss": loss,
                                }
                            else:
                                print(f"Duplicate suffix {suffix} for mode {mode} in {model_run_name}. Skipping.")

                except (json.JSONDecodeError, FileNotFoundError) as e:
                    print(f"Error reading {file_path}: {e}")
                    raise e

    print("Finished processing model runs.")
    return data



def process_runs() -> dict:
    base_dir = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/trainclassif/sweep_res/"
    runs = [d for d in os.listdir(base_dir) if os.path.isdir(os.path.join(base_dir, d))]
    models = ["efficientnet_b2", "resnet50", "vit_b_16"]

    model_res = {model: {} for model in models}
    results = {run: deepcopy(model_res) for run in runs}
    for run in runs:
        for model in models:
            model_results_path = os.path.join(base_dir, run, f"{model}_13")
            data = process_model_dataset_runs(model_results_path)
            results[run][model] = data

    return results


if __name__ == "__main__":
    run_results = process_runs()

    res_path = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/trainclassif/sweep_res/run_summary.json"
    with open(res_path, "w") as f:
        json.dump(run_results, f, indent=2)