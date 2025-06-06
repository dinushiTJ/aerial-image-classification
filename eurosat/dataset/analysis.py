import json
from datasets import load_dataset, get_dataset_infos
from collections import Counter
from typing import Dict


DATASET_NAME = "blanchon/EuroSAT_RGB"


def print_dataset_info(dataset_name: str) -> None:
    ds_infos = get_dataset_infos(dataset_name)
    ds_info = next(iter(ds_infos.values()))
    features = ds_info.features
    num_classes = features["label"].num_classes
    class_names = features["label"].names
    id2label = {i: name for i, name in enumerate(class_names)}
    label2id = {name: i for i, name in enumerate(class_names)}
    
    print(f"Number of classes: {num_classes}")
    print(f"Class names: {class_names}\n")

    ds = load_dataset(dataset_name)

    data_counts = dict()
    for split in ds.keys():
        labels = ds[split]["label"]
        counter: Dict[int, int] = Counter(labels)
        split_data_counts: dict[str, int] = dict()

        for class_id in range(num_classes):
            class_name = id2label[class_id]
            count = counter.get(class_id, 0)
            split_data_counts[class_name] = count

        data_counts[split] = dict(sorted(split_data_counts.items(), key=lambda x: x[1]))

    print(json.dumps(data_counts, indent=2))



if __name__ == "__main__":
    print_dataset_info(DATASET_NAME)
