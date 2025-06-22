import random
import matplotlib
matplotlib.use('Agg')  # Use non-GUI backend
import matplotlib.pyplot as plt
from datasets import load_dataset

def save_random_class_grid_svg(
    dataset_name: str,
    output_path: str = "class_grid.svg",
    split: str = "train",
    image_key: str = "image",
    label_key: str = "label",
    num_classes: int = 13,
    seed: int = 42
):
    """
    Saves a 3x5 grid of randomly selected images (one per class) as an SVG file.
    """
    random.seed(seed)

    dataset = load_dataset(dataset_name, split=split)
    label_names = dataset.features[label_key].names

    assert len(label_names) >= num_classes, f"Expected at least {num_classes} classes."

    # Collect one random image per class
    samples = []
    for class_id in range(num_classes):
        items = [item for item in dataset if item[label_key] == class_id]
        if not items:
            raise ValueError(f"No samples found for class {class_id}")
        sample = random.choice(items)
        samples.append((sample[image_key], label_names[class_id]))

    # Create 3x5 grid
    fig, axes = plt.subplots(3, 5, figsize=(15, 9))
    axes = axes.flatten()

    for ax, (img, label) in zip(axes, samples):
        ax.imshow(img)
        ax.set_title(label, fontsize=10)
        ax.axis("off")

    for i in range(num_classes, len(axes)):
        axes[i].axis("off")

    plt.tight_layout()
    plt.savefig(output_path, format='svg')
    plt.close()


if __name__ == "__main__":
    output_svg_path = "/home/dj191/research/code/waikato_aerial/dataset/plots"
    
    for steps in [25, 50]:
        save_random_class_grid_svg(
            dataset_name=f"dushj98/aerial_synthetic_base_{steps}",
            output_path=f"{output_svg_path}/sd_base_{steps}.svg",
            seed=42,
        )
