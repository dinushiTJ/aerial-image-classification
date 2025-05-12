import os
from datasets import DatasetDict, load_dataset, ClassLabel
from dotenv import load_dotenv
from huggingface_hub import login

load_dotenv("../.env")
HF_TOKEN = os.environ.get("HF_TOKEN")

login(token=HF_TOKEN)


def prep_new_ds() -> None:
    ds = load_dataset("dushj98/waikato_aerial_imagery_2017")

    classes_to_remove = [
        "broadleaved_indigenous_hardwood",
        "grose_broom",
        "low_producing_grassland",
        "urban_parkland",
        "herbaceous_freshwater_vege",
        "manuka_kanuka"
    ]

    original_label_feature = ds["train"].features["label"]
    original_names = original_label_feature.names

    # New class list
    new_class_names = [name for name in original_names if name not in classes_to_remove]
    name2new_idx = {name: i for i, name in enumerate(new_class_names)}
    idx2name = {i: name for i, name in enumerate(original_names)}

    def is_valid_class(example):
        return idx2name[example["label"]] in name2new_idx

    def remap_label(example):
        class_name = idx2name[example["label"]]
        example["label"] = name2new_idx[class_name]
        return example

    new_ds = DatasetDict()
    for split in ds:
        filtered = ds[split].filter(is_valid_class)
        remapped = filtered.map(remap_label)
        remapped = remapped.cast_column("label", ClassLabel(names=new_class_names))
        new_ds[split] = remapped

    new_repo_id = "dushj98/waikato_aerial_imagery_2017_7cls"
    new_ds.push_to_hub(new_repo_id)
    print(f"Pushed filtered dataset to https://huggingface.co/datasets/{new_repo_id}")


if __name__ == "__main__":
    prep_new_ds()
