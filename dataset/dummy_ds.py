import click
from datasets import load_dataset, Dataset, DatasetDict, Features, ClassLabel
from huggingface_hub import login
from collections import defaultdict

@click.command()
@click.option('--source_dataset', '-s', type=str, default='dushj98/waikato_aerial_imagery_2017',
              help='Source dataset on Hugging Face')
@click.option('--output_name', '-o', type=str, default='dushj98/aerial-imagery-mini',
              help='Name for the output dataset on Hugging Face')
@click.option('--hf_token', '-h', type=str, required=True, help='Hugging Face token')
@click.option('--samples_per_class', '-n', type=int, default=2,
              help='Number of samples per class for both training and validation')
def create(source_dataset, output_name, hf_token, samples_per_class):
    login(token=hf_token)
    print(f"🔍 Loading dataset: {source_dataset}")
    dataset = load_dataset(source_dataset)
    print(f"✅ Splits found: {list(dataset.keys())}")

    label_feature = dataset["train"].features["label"]
    if not isinstance(label_feature, ClassLabel):
        raise ValueError("❌ Expected 'label' to be a ClassLabel")

    num_classes = label_feature.num_classes
    label_names = label_feature.names
    print(f"🧠 Found {num_classes} classes: {label_names}")

    def sample_split_fast(split_name):
        split_data = dataset[split_name].shuffle(seed=42)
        class_buckets = defaultdict(list)

        for example in split_data:
            label = example["label"]
            if len(class_buckets[label]) < samples_per_class:
                class_buckets[label].append(example)
            if all(len(class_buckets[i]) >= samples_per_class for i in range(num_classes)):
                break

        flat = [item for sublist in class_buckets.values() for item in sublist]
        ds = Dataset.from_list(flat)

        # Re-apply original features (especially for ClassLabel restoration)
        ds = ds.cast(dataset[split_name].features)
        return ds

    mini_train = sample_split_fast("train")
    mini_val = sample_split_fast("validation")

    mini_ds = DatasetDict({
        "train": mini_train,
        "validation": mini_val
    })

    print(f"📊 Mini dataset: train={len(mini_train)}, val={len(mini_val)}")
    mini_ds.push_to_hub(output_name)
    print(f"✅ Uploaded to HF: {output_name}")

if __name__ == "__main__":
    create()
