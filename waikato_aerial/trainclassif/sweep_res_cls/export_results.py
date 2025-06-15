from copy import deepcopy
import json
import os


seeds = [1417352920, 319080682, 3892354109, 34793895, 42]
classes = ['broadleaved_indigenous_hardwood', 'deciduous_hardwood', 'grose_broom', 'harvested_forest', 'herbaceous_freshwater_vege', 'high_producing_grassland', 'indigenous_forest', 'lake_pond', 'low_producing_grassland', 'manuka_kanuka', 'shortrotation_cropland', 'urban_build_up', 'urban_parkland']
models = ["efficientnet", "resnet50", "vit"]
training_modes = ['tl', 'sft', 'fft']
suffixes = ['real', 'p10', 'p25', 'p50', 'p75', 'p100', 'p125', 'p150']


def get_metadata(file_name: str) -> tuple[str, str, str]:
    model = None
    training_mode = None
    dataset = None

    for m in models:
        if m in file_name:
            model = m
            break
    
    if not model:
        raise ValueError(f"Model not found in file name: {file_name}")
    
    for mode in training_modes:
        if f"-{mode}-" in file_name or file_name.endswith(f"-{mode}.json"):
            training_mode = mode
            break
    
    if not training_mode:
        raise ValueError(f"Training mode not found in file name: {file_name}")
    
    for d in suffixes:
        if f"13cls-{d}-" in file_name:
            dataset = d
            break
    
    if not dataset:
        raise ValueError(f"Dataset suffix not found in file name: {file_name}")
    
    return model, training_mode, dataset


def process_runs(base_dir: str) -> dict:
    print("Processing class runs...")

    dir_list = []
    for seed in seeds:
        seed_res_dir = os.path.join(base_dir, f"seed{seed}")

        for model in models:
            seed_model_path = os.path.join(seed_res_dir, model)
            dir_list.append(seed_model_path)
    
    print(f"Found {len(dir_list)} directories to process.")
    print(f"---\nDirectories: \n{'\n'.join(dir_list[:2])}\n---")

    all_files_to_process = []
    for d in dir_list:
        # read Json files in the directory
        file_list = [f for f in os.listdir(d) if f.endswith('.json')]
        all_files_to_process.extend([os.path.join(d, f) for f in file_list])
    
    all_files_to_process = set(all_files_to_process)
    print(f"Found {len(all_files_to_process)} files to process.")
    print(f"---\nFiles:")
    print("\n".join([os.path.basename(f) for f in all_files_to_process if ("vit_b_16-13cls-p25" in f or "vit-b-16-13cls-p25" in f)]))
    print("---")
    
    cls_res = {c: {} for c in classes}
    for f in all_files_to_process:
        base_f = str(os.path.basename(f))
        
        # # Just for testing
        # if not base_f.startswith("vit_b_16-13cls-p25") and not base_f.startswith("vit-b-16-13cls-p25"):
        #     print(f"Skipping file: {base_f} as 'vit_b_16-13cls-p25' or 'vit-b-16-13cls-p25' not in the name.")
        #     continue
            
        print(f"Processing file: {base_f}")

        with open(f, "r") as file:
            data = json.load(file)
        
        model, training_mode, dataset = get_metadata(os.path.basename(f))
        cls_data = data["class"]
        for cls, cls_metrics in cls_data.items():
            if cls not in cls_res:
                cls_res[cls] = {}

            if model not in cls_res[cls]:
                cls_res[cls][model] = {}

            if training_mode not in cls_res[cls][model]:
                cls_res[cls][model][training_mode] = {}
        
            if dataset not in cls_res[cls][model][training_mode]:
                cls_res[cls][model][training_mode][dataset] = {
                    "accuracy": [],
                    "precision": [],
                    "recall": [],
                    "f1_score": [],
                }
        
            cls_res[cls][model][training_mode][dataset]["accuracy"].append(cls_metrics["accuracy"])
            cls_res[cls][model][training_mode][dataset]["precision"].append(cls_metrics["precision"])
            cls_res[cls][model][training_mode][dataset]["recall"].append(cls_metrics["recall"])
            cls_res[cls][model][training_mode][dataset]["f1_score"].append(cls_metrics["f1"])
        
        print()

    # peek
    print(json.dumps(cls_res, indent=2))

    # verify
    for cls, model_res in cls_res.items():
        for model, training_mode_res in model_res.items():
            for training_mode, dataset_res in training_mode_res.items():
                for dataset, metrics in dataset_res.items():
                    if len(metrics["accuracy"]) != 5 or len(metrics["precision"]) != 5 or len(metrics["recall"]) != 5 or len(metrics["f1_score"]) != 5:
                        raise Exception(f"Warning: Incomplete data for {cls}, {model}, {training_mode}, {dataset}. Expected 5 entries, got {len(metrics['accuracy'])} accuracy, {len(metrics['precision'])} precision, {len(metrics['recall'])} recall and {len(metrics['f1_score'])} f1_score entries.")

    print("Processing complete.")
    return cls_res


if __name__ == "__main__":
    base_dir = "C:/Users/arcad/Downloads/d/repo/aerial-image-classification/waikato_aerial/trainclassif/sweep_res_cls/"
    cls_res = process_runs(base_dir=base_dir)

    res_path = f"{base_dir}run_summary_cls.json"
    with open(res_path, "w") as f:
        json.dump(cls_res, f, indent=2)