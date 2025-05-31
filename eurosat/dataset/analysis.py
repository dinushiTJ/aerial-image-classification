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

    for split in ds.keys():
        print(f"Split: {split}")
        labels = ds[split]["label"]
        counter: Dict[int, int] = Counter(labels)
        for class_id in range(num_classes):
            class_name = id2label[class_id]
            count = counter.get(class_id, 0)
            print(f"  {class_name}: {count} images")
        print()


if __name__ == "__main__":
    print_dataset_info(DATASET_NAME)
