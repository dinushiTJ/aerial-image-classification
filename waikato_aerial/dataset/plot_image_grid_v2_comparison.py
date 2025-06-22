import random
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from datasets import load_dataset


def save_class_samples_grid_svg(
    dataset_name: str,
    class_name: str,
    output_path: str,
    caption: str,
    samples: int = 4,
    split: str = "train",
    image_key: str = "image",
    label_key: str = "label",
    seed: int = 42
):
    assert samples == 4, "This function is configured to show exactly 4 samples (2x2)."

    random.seed(seed)
    dataset = load_dataset(dataset_name, split=split)
    label_names = dataset.features[label_key].names

    # Get class ID
    if class_name not in label_names:
        raise ValueError(f"Class name '{class_name}' not found in label names.")
    class_id = label_names.index(class_name)

    # Filter items with the target class
    class_items = [item for item in dataset if item[label_key] == class_id]
    if len(class_items) < samples:
        raise ValueError(f"Not enough samples for class '{class_name}'. Found {len(class_items)}.")

    selected = random.sample(class_items, samples)

    # Create 2x2 grid
    fig, axes = plt.subplots(2, 2, figsize=(6, 6))
    axes = axes.flatten()

    for ax, sample in zip(axes, selected):
        ax.imshow(sample[image_key])
        ax.axis("off")

    # Add caption under the figure
    fig.suptitle(caption, fontsize=12, y=0.05)
    plt.subplots_adjust(top=0.95, bottom=0.15)

    plt.savefig(output_path, format='svg')
    plt.close()


if __name__ == "__main__":
    output_svg_path = "/home/dj191/research/code/waikato_aerial/dataset/plots/ds_comparison"
    class_name = "broadleaved_indigenous_hardwood"

    datasets = {
        "dushj98/waikato_aerial_imagery_2017", "Ground truth images of {class_name} from Waikato Aerial 2017",
        "dushj98/aerial_synthetic_base_50", "Synthetic images of {class_name} from direct Stable Diffusion inference",
        "dushj98/waikato_aerial_2017_synthetic_best_cmmd", "Synthetic images of {class_name} from fine-tuned Stable Diffusion inference",
    }
    samples = 4

    for ds, caption in enumerate(datasets):
        ds_name = ds.split("/")[-1]
        save_class_samples_grid_svg(
            dataset_name=ds,
            output_path=f"{output_svg_path}/{ds_name}_{class_name}_sample.svg",
            class_name=class_name,
            samples=samples,
            caption=caption,
            seed=42,
        )
