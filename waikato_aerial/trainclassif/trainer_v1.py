import click
import wandb
import torch
from torch import device, max, set_grad_enabled
from torch.nn import Module, Linear, CrossEntropyLoss
from torch.optim import Adam, Optimizer
from torch.cuda import is_available as is_cuda_available
from torch.utils.data import DataLoader
from torchvision import models, transforms
from datasets import load_dataset
from sklearn.metrics import precision_score, recall_score, f1_score
import timm
import copy
import time
import os
import random
import numpy as np


model_configs = {
    "mobilenet_v2": {"batch_size": 64, "input_size": 224},
    "resnet50": {"batch_size": 32, "input_size": 224},
    "efficientnet_b2": {"batch_size": 32, "input_size": 224},
    "efficientnetv2_m": {"batch_size": 16, "input_size": 224},
    "vit_b_16": {"batch_size": 16, "input_size": 224},
}


def set_reproducibility(seed=42):
    """
    Set all random seeds and deterministic flags to ensure reproducibility.

    Args:
        seed (int): The seed value to use for reproducibility
    """
    # Set Python random seed
    random.seed(seed)

    # Set NumPy random seed
    np.random.seed(seed)

    # Set PyTorch random seeds
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)  # For multi-GPU

    # Set deterministic behavior
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False

    # For DataLoader reproducibility
    def seed_worker(worker_id):
        worker_seed = seed + worker_id
        random.seed(worker_seed)
        np.random.seed(worker_seed)
        torch.manual_seed(worker_seed)

    return seed_worker


def load_data(
    dataset_name: str, input_size: int, batch_size: int, seed_worker=None, g=None
) -> dict[str, DataLoader]:
    dataset = load_dataset(dataset_name)
    
    # Define the transform
    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def transform_dataset(example):
        # Convert the image to RGB if it has alpha channel or other formats
        if hasattr(example["image"], 'convert'):
            example["image"] = example["image"].convert("RGB")
        example["image"] = transform(example["image"])
        return example

    # Apply transformations
    dataset = dataset.map(transform_dataset)
    
    # Set the format to PyTorch tensors
    dataset.set_format(type='torch', columns=['image', 'label'])

    dataloaders = {
        "train": DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g,
        ),
        "val": DataLoader(
            dataset["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=4,
            worker_init_fn=seed_worker,
            generator=g,
        ),
    }

    return dataloaders


def set_parameter_requires_grad(model: Module, feature_extracting: bool) -> None:
    for param in model.parameters():
        param.requires_grad = not feature_extracting


def initialize_model(
    model_name: str,
    num_classes: int,
    feature_extract: bool,
    use_pretrained: bool = True,
) -> tuple[Module, int]:
    model_ft = None
    input_size = model_configs[model_name]["input_size"]

    if model_name == "mobilenet_v2":
        model_ft = models.mobilenet_v2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
        model_ft = models.resnet50(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = Linear(num_ftrs, num_classes)
    elif model_name == "efficientnet_b2":
        model_ft = models.efficientnet_b2(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = Linear(num_ftrs, num_classes)
    elif model_name == "efficientnetv2_m":
        model_ft = models.efficientnet_v2_m(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = Linear(num_ftrs, num_classes)
    elif model_name == "vit_b_16":
        model_ft = models.vit_b_16(pretrained=use_pretrained)
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.heads.head.in_features
        model_ft.heads.head = Linear(num_ftrs, num_classes)
    else:
        raise ValueError(f"Invalid model name: {model_name}")

    # Ensure the final layer's parameters require gradients
    if model_name == "mobilenet_v2":
        model_ft.classifier[1].weight.requires_grad = True
        model_ft.classifier[1].bias.requires_grad = True
    elif model_name == "resnet50":
        model_ft.fc.weight.requires_grad = True
        model_ft.fc.bias.requires_grad = True
    elif model_name == "efficientnet_b2":
        model_ft.classifier[1].weight.requires_grad = True
        model_ft.classifier[1].bias.requires_grad = True
    elif model_name == "vit_b_16":
        model_ft.heads.head.weight.requires_grad = True
        model_ft.heads.head.bias.requires_grad = True

    return model_ft, input_size


def train_model(
    model: Module,
    dataloaders: dict[str, DataLoader],
    criterion: Module,
    optimizer: Optimizer,
    num_epochs: int = 25,
    patience: int = 5,
    wandb_log: bool = False,
    model_name: str = "model",
) -> tuple[Module, dict]:
    since = time.perf_counter()
    val_acc_history = []
    best_epoch, best_acc, best_loss = 0, 0.0, float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())
    checkpoint_path = f"best_checkpoint_{model_name}.pt"
    results = {
        "best_val_acc": 0.0,
        "best_val_loss": float("inf"),
        "best_epoch": 0,
        "val_acc_history": [],
        "train_time": 0.0,
    }

    dev = device("cuda:0" if is_cuda_available() else "cpu")
    model.to(dev)

    for epoch in range(num_epochs):
        print(f"Epoch {epoch}/{num_epochs - 1}\n" + "-" * 10)
        epoch_metrics = {}

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0.0, 0
            all_preds, all_labels = [], []

            for batch in dataloaders[phase]:
                inputs, labels = batch["image"].to(dev), batch["label"].to(dev)
                
                optimizer.zero_grad()
                with set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = max(outputs, 1)[1]
                    if phase == "train":
                        loss.backward()
                        optimizer.step()
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            epoch_metrics[f"{phase}_loss"] = epoch_loss
            epoch_metrics[f"{phase}_acc"] = epoch_acc

            if phase == "val":
                # Calculate additional metrics
                epoch_metrics["val_precision"] = precision_score(
                    all_labels, all_preds, average="macro", zero_division=0
                )
                epoch_metrics["val_recall"] = recall_score(
                    all_labels, all_preds, average="macro"
                )
                epoch_metrics["val_f1"] = f1_score(
                    all_labels, all_preds, average="macro"
                )

                val_acc_history.append(epoch_acc)

                if epoch_acc > best_acc:
                    best_acc, best_loss, best_epoch = epoch_acc, epoch_loss, epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Save the best checkpoint
                    if wandb_log:
                        try:
                            torch.save(model.state_dict(), checkpoint_path)
                            wandb.save(checkpoint_path)
                            print(
                                f"Saved best checkpoint at epoch {epoch} with accuracy {epoch_acc:.4f}"
                            )
                        except Exception as e:
                            print(f"Failed to save checkpoint: {e}")

            print(f"{phase} Loss: {epoch_loss:.4f} Acc: {epoch_acc:.4f}")

        if wandb_log:
            try:
                wandb.log(epoch_metrics, step=epoch)
            except Exception as e:
                print(f"WandB logging failed: {e}")

        # Early stopping
        if best_epoch < epoch - patience:
            print(f"Early stopping at epoch {epoch}")
            break
        print()

    time_elapsed = time.perf_counter() - since
    print(f"Training complete in {time_elapsed // 60:.0f}m {time_elapsed % 60:.0f}s")
    print(f"Best val Acc: {best_acc:.4f} at epoch {best_epoch}")

    # Load best model weights
    model.load_state_dict(best_model_wts)

    # Save results
    results["best_val_acc"] = best_acc
    results["best_val_loss"] = best_loss
    results["best_epoch"] = best_epoch
    results["val_acc_history"] = val_acc_history
    results["train_time"] = time_elapsed

    return model, results


@click.command()
@click.option(
    "--dataset_name", "-d", type=str, required=True, help="Hugging Face dataset name"
)
@click.option("--hf_token", "-h", type=str, required=True, help="Hugging Face token")
@click.option(
    "--wandb_token", "-w", type=str, required=True, help="Weights & Biases token"
)
@click.option("--epochs", "-e", type=int, default=25, help="Number of training epochs")
@click.option("--lr", "-l", type=float, default=1e-3, help="Learning rate")
@click.option(
    "--use_wandb", "-wb", is_flag=True, help="Enable Weights & Biases logging"
)
@click.option(
    "--metadata",
    "-m",
    type=str,
    default="",
    help="Optional metadata as key1=value1,key2=value2",
)
@click.option(
    "--models",
    "-md",
    type=str,
    default="all",
    help='Comma-separated list of models to train (or "all")',
)
@click.option(
    "--seed", "-s", type=int, default=42, help="Random seed for reproducibility"
)
def train(
    dataset_name: str,
    hf_token: str,
    wandb_token: str,
    epochs: int,
    lr: float,
    use_wandb: bool,
    metadata: str,
    models: str,
    seed: int,
):
    # Set reproducibility
    print(f"Setting random seed to {seed} for reproducibility")
    seed_worker = set_reproducibility(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # Setup Hugging Face authentication
    try:
        from huggingface_hub import login

        login(token=hf_token)
        print("Successfully logged in to Hugging Face Hub")
    except Exception as e:
        print(f"Failed to login to Hugging Face Hub: {e}")
        return

    # Setup Weights & Biases
    if use_wandb:
        try:
            import wandb

            wandb.login(key=wandb_token)
            print("Successfully logged in to Weights & Biases")
        except Exception as e:
            print(f"Failed to login to Weights & Biases: {e}")

    # Parse metadata
    metadata_dict = {}
    if metadata:
        try:
            for item in metadata.split(","):
                if "=" in item:
                    key, value = item.split("=", 1)
                    metadata_dict[key.strip()] = value.strip()
        except Exception as e:
            print(f"Failed to parse metadata: {e}")
            metadata_dict["raw_metadata"] = metadata

    # Add reproducibility info to metadata
    metadata_dict["random_seed"] = seed

    # Determine which models to train
    if models.lower() == "all":
        models_to_train = list(model_configs.keys())
    else:
        models_to_train = [m.strip() for m in models.split(",")]
        # Validate models
        for model in models_to_train:
            if model not in model_configs:
                print(f"Warning: Model '{model}' not recognized. Skipping.")
                models_to_train.remove(model)
                
    print(f"Will train the following models: {', '.join(models_to_train)}")

    # Must match the classes 
    # in the dataset
    num_classes = 13

    # Train each model sequentially
    all_results = {}
    for model_name in models_to_train:
        print(f"\n{'=' * 50}")
        print(f"Training {model_name} on {dataset_name}")
        print(f"{'=' * 50}")

        # End any existing W&B run
        if use_wandb and wandb.run is not None:
            wandb.finish()

        # Initialize W&B run with config for this model
        if use_wandb:
            config = {
                "model": model_name,
                "epochs": epochs,
                "lr": lr,
                "batch_size": model_configs[model_name]["batch_size"],
                "dataset": dataset_name,
                "random_seed": seed,  # Log the seed in the config
                **metadata_dict,
            }

            run_name = (
                f"{model_name}_{dataset_name}_seed{seed}"  # Include seed in run name
            )
            try:
                wandb.init(project="model-training", name=run_name, config=config)
                print(f"W&B run initialized for {model_name} with config: {config}")
            except Exception as e:
                print(f"Failed to initialize W&B run: {e}")

        # Initialize model
        model_ft, input_size = initialize_model(
            model_name, num_classes, feature_extract=True, use_pretrained=True
        )

        # Load and prepare dataset with reproducibility
        batch_size = model_configs[model_name]["batch_size"]
        dataloaders_dict = load_data(
            dataset_name, input_size, batch_size, seed_worker, g
        )

        # Set up loss function and optimizer
        criterion = CrossEntropyLoss()
        optimizer = Adam(model_ft.parameters(), lr=lr)

        # Train the model
        _, results = train_model(
            model_ft,
            dataloaders_dict,
            criterion,
            optimizer,
            num_epochs=epochs,
            wandb_log=use_wandb,
            model_name=f"{model_name}_seed{seed}",  # Include seed in model name
        )

        # Store results
        all_results[model_name] = results

        # Make sure to close the wandb run
        if use_wandb and wandb.run is not None:
            wandb.finish()

    # Print summary of results
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Dataset: {dataset_name}")
    print(f"Random Seed: {seed}")  # Print the seed used

    # Create a table of results
    header = (
        f"{'Model':<15} | {'Best Acc':<10} | {'Best Epoch':<10} | {'Train Time':<10}"
    )
    print("\n" + header)
    print("-" * len(header))

    for model_name, results in all_results.items():
        acc = results["best_val_acc"] * 100
        epoch = results["best_epoch"]
        time_mins = results["train_time"] / 60
        print(f"{model_name:<15} | {acc:>9.2f}% | {epoch:>10d} | {time_mins:>9.1f}m")

    # Save results with seed for reproducibility tracking
    results_filename = f"training_results_seed{seed}.txt"
    try:
        with open(results_filename, "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Random Seed: {seed}\n\n")
            f.write(header + "\n")
            f.write("-" * len(header) + "\n")
            for model_name, results in all_results.items():
                acc = results["best_val_acc"] * 100
                epoch = results["best_epoch"]
                time_mins = results["train_time"] / 60
                f.write(
                    f"{model_name:<15} | {acc:>9.2f}% | {epoch:>10d} | {time_mins:>9.1f}m\n"
                )
        print(f"\nResults saved to {results_filename}")
    except Exception as e:
        print(f"Failed to save results to file: {e}")


if __name__ == "__main__":
    train()
