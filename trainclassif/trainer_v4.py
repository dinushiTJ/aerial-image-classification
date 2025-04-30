import copy
import random
import time

import click
import numpy as np
import torch
import wandb
from datasets import load_dataset
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from torch import device
from torch import max
from torch import set_grad_enabled
from torch.cuda import is_available as is_cuda_available
from torch.nn import CrossEntropyLoss
from torch.nn import Linear
from torch.nn import Module
from torch.optim import Adam
from torch.optim import Optimizer
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from tqdm.auto import tqdm

training_mode_map = {
    "tl": "Transfer Learning",
    "ft": "Fine-tuning",
    "pt": "Pre-training",
}

model_configs = {
    "mobilenet_v2": {"batch_size": 64, "input_size": 224},
    "resnet50": {"batch_size": 32, "input_size": 224},
    "efficientnet_b2": {"batch_size": 32, "input_size": 224},
    "efficientnetv2_m": {"batch_size": 16, "input_size": 224},
    "vit_b_16": {"batch_size": 16, "input_size": 224},
}


def set_reproducibility(seed=42) -> None:
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

    transform = transforms.Compose(
        [
            transforms.Resize(256),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            transforms.Normalize([0.485, 0.456, 0.406], [0.229, 0.224, 0.225]),
        ]
    )

    def transform_dataset(example):
        if hasattr(example["image"], "convert"):
            example["image"] = example["image"].convert("RGB")
        example["image"] = transform(example["image"])
        return example

    # Apply transformations
    dataset = dataset.map(transform_dataset)

    # Set the format to PyTorch tensors
    dataset.set_format(type="torch", columns=["image", "label"])

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
        model_ft = models.mobilenet_v2(
            weights=(
                models.MobileNet_V2_Weights.IMAGENET1K_V2 if use_pretrained else None
            )
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = Linear(num_ftrs, num_classes)
    elif model_name == "resnet50":
        model_ft = models.resnet50(
            weights=models.ResNet50_Weights.IMAGENET1K_V2 if use_pretrained else None
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.fc.in_features
        model_ft.fc = Linear(num_ftrs, num_classes)
    elif model_name == "efficientnet_b2":
        model_ft = models.efficientnet_b2(
            weights=(
                models.EfficientNet_B2_Weights.IMAGENET1K_V1 if use_pretrained else None
            )
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = Linear(num_ftrs, num_classes)
    elif model_name == "efficientnetv2_m":
        model_ft = models.efficientnet_v2_m(
            weights=(
                models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
                if use_pretrained
                else None
            )
        )
        set_parameter_requires_grad(model_ft, feature_extract)
        num_ftrs = model_ft.classifier[1].in_features
        model_ft.classifier[1] = Linear(num_ftrs, num_classes)
    elif model_name == "vit_b_16":
        model_ft = models.vit_b_16(
            weights=models.ViT_B_16_Weights.IMAGENET1K_V1 if use_pretrained else None
        )
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
    elif model_name == "efficientnetv2_m":
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

    # Create epoch progress bar with TQDM
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0)

    for epoch in epoch_pbar:
        epoch_metrics = {}

        for phase in ["train", "val"]:
            model.train() if phase == "train" else model.eval()
            running_loss, running_corrects = 0.0, 0
            all_preds, all_labels = [], []

            # Create batch progress bar
            batch_pbar = tqdm(
                dataloaders[phase],
                desc=f"{phase.capitalize()}",
                position=1,
                leave=False,
                total=len(dataloaders[phase]),
            )

            for batch in batch_pbar:
                inputs, labels = batch["image"].to(dev), batch["label"].to(dev)

                optimizer.zero_grad()
                with set_grad_enabled(phase == "train"):
                    outputs = model(inputs)
                    loss = criterion(outputs, labels)
                    preds = max(outputs, 1)[1]
                    if phase == "train":
                        loss.backward()
                        optimizer.step()

                # Update metrics
                running_loss += loss.item() * inputs.size(0)
                running_corrects += (preds == labels).sum().item()
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update batch progress bar
                batch_pbar.set_postfix({"loss": f"{loss.item():.4f}"})

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            epoch_metrics[f"{phase}_loss"] = epoch_loss
            epoch_metrics[f"{phase}_acc"] = epoch_acc

            # Calculate precision, recall, and F1 for both train and val phases
            epoch_metrics[f"{phase}_precision"] = precision_score(
                all_labels, all_preds, average="macro", zero_division=0
            )
            epoch_metrics[f"{phase}_recall"] = recall_score(
                all_labels, all_preds, average="macro", zero_division=0
            )
            epoch_metrics[f"{phase}_f1"] = f1_score(
                all_labels, all_preds, average="macro", zero_division=0
            )

            if phase == "val":
                val_acc_history.append(epoch_acc)

                if epoch_acc > best_acc:
                    best_acc, best_loss, best_epoch = epoch_acc, epoch_loss, epoch
                    best_model_wts = copy.deepcopy(model.state_dict())

                    # Save the best checkpoint
                    try:
                        torch.save(model.state_dict(), checkpoint_path)
                        wandb.save(checkpoint_path)
                        tqdm.write(
                            f"Saved best checkpoint at epoch {epoch} with accuracy {epoch_acc:.4f}"
                        )
                    except Exception as e:
                        tqdm.write(f"Failed to save checkpoint: {e}")

            # Update epoch progress bar with the metrics
            postfix = {
                f"{phase}_loss": f"{epoch_loss:.4f}",
                f"{phase}_acc": f"{epoch_acc:.4f}",
            }

            # For the validation phase, also show if this is the best model so far
            if phase == "val":
                postfix["best_acc"] = f"{best_acc:.4f}"
                # Update the epoch progress bar
                epoch_pbar.set_postfix(postfix)
                tqdm.write(
                    f"Epoch {epoch}/{num_epochs-1} - Train loss: {epoch_metrics['train_loss']:.4f}, "
                    f"acc: {epoch_metrics['train_acc']:.4f} | Val loss: {epoch_loss:.4f}, "
                    f"acc: {epoch_acc:.4f} | "
                    f"Train F1: {epoch_metrics['train_f1']:.4f}, Val F1: {epoch_metrics['val_f1']:.4f}"
                )

        try:
            wandb.log(epoch_metrics, step=epoch)
        except Exception as e:
            tqdm.write(f"WandB logging failed: {e}")

        # Early stopping
        if best_epoch < epoch - patience:
            tqdm.write(f"Early stopping at epoch {epoch}")
            break

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

    # Save the best model
    torch.save(model.state_dict(), checkpoint_path)

    return model, results, checkpoint_path


def get_architecture_class_name(model_name):
    """Get the class name for the architecture based on model name."""
    mapping = {
        "mobilenet_v2": "MobileNetV2ForImageClassification",
        "resnet50": "ResNetForImageClassification",
        "efficientnet_b2": "EfficientNetForImageClassification",
        "efficientnetv2_m": "EfficientNetV2ForImageClassification",
        "vit_b_16": "ViTForImageClassification",
    }
    return mapping.get(model_name, "ImageClassificationModel")


def get_model_type(model_name):
    """Get the model type for the config."""
    mapping = {
        "mobilenet_v2": "mobilenet_v2",
        "resnet50": "resnet",
        "efficientnet_b2": "efficientnet",
        "efficientnetv2_m": "efficientnet_v2",
        "vit_b_16": "vit",
    }
    return mapping.get(model_name, model_name.replace("_", "-"))


def upload_model_to_hf(
    model_name: str,
    dataset_name: str,
    hf_repo_id: str,
    checkpoint_path: str,
    num_classes: int,
    metadata: dict = None,
) -> None:
    """
    Upload the model to HuggingFace Model Hub.

    Args:
        model: The trained model instance
        model_name: Name of the model architecture
        dataset_name: Name of the dataset used for training
        hf_repo_id: HuggingFace repository ID (e.g., "dushj98/model-name")
        checkpoint_path: Path to the saved model checkpoint
        num_classes: Number of classes in the dataset
        metadata: Additional metadata to include in the model card
    """
    try:
        import json
        import os

        from huggingface_hub import HfApi
        from huggingface_hub import ModelCard
        from huggingface_hub import create_repo

        # Create repo if it doesn't exist
        try:
            create_repo(hf_repo_id, exist_ok=True)
            print(f"Repository '{hf_repo_id}' is ready")
        except Exception as e:
            print(f"Error creating repository: {e}")
            return False

        # Create simple model card with metadata
        model_card_content = f"""
# {model_name} trained on {dataset_name}

This model is a fine-tuned version of {model_name} on the {dataset_name} dataset.

## Model Details

- Model Type: {model_name}
- Training Dataset: {dataset_name}
- Number of classes: {num_classes}
        """

        if metadata:
            model_card_content += "\n## Training Metadata\n\n"
            for key, value in metadata.items():
                model_card_content += f"- {key}: {value}\n"

        model_card = ModelCard(model_card_content)

        # Create config.json - simplified approach
        config = {
            "architectures": [get_architecture_class_name(model_name)],
            "model_type": get_model_type(model_name),
            "num_labels": num_classes,
            "id2label": {str(i): f"class_{i}" for i in range(num_classes)},
            "label2id": {f"class_{i}": i for i in range(num_classes)},
            "image_size": model_configs[model_name]["input_size"],
            "num_channels": 3,
        }

        with open("config.json", "w") as f:
            json.dump(config, f, indent=2)

        # Create model_index.json
        model_index = {
            "name": f"{model_name} for image classification",
            "dataset": dataset_name,
            "tags": [
                "image-classification",
                model_name,
                dataset_name.replace("/", "-"),
            ],
            "task": "image-classification",
            "library_name": "pytorch",
            "pipeline_tag": "image-classification",
        }

        with open("model_index.json", "w") as f:
            json.dump(model_index, f, indent=2)

        # Create preprocessor_config.json for vision models
        preprocessor_config = {
            "do_normalize": True,
            "do_resize": True,
            "image_mean": [0.485, 0.456, 0.406],
            "image_std": [0.229, 0.224, 0.225],
            "size": {
                "height": model_configs[model_name]["input_size"],
                "width": model_configs[model_name]["input_size"],
            },
        }

        with open("preprocessor_config.json", "w") as f:
            json.dump(preprocessor_config, f, indent=2)

        # Init API
        api = HfApi()

        # Upload model weights
        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo=f"pytorch_model.bin",
            repo_id=hf_repo_id,
            commit_message=f"Upload {model_name} trained on {dataset_name}",
        )

        # Upload config files
        for config_file in [
            "config.json",
            "model_index.json",
            "preprocessor_config.json",
        ]:
            api.upload_file(
                path_or_fileobj=config_file,
                path_in_repo=config_file,
                repo_id=hf_repo_id,
                commit_message=f"Upload {config_file}",
            )
            # Clean up local file
            os.remove(config_file)

        # Save and upload model card
        model_card.save(f"model_card.md")
        api.upload_file(
            path_or_fileobj="model_card.md",
            path_in_repo="README.md",
            repo_id=hf_repo_id,
            commit_message="Upload model card",
        )
        os.remove("model_card.md")

        return True

    except Exception as e:
        print(f"Failed to upload model to HuggingFace: {e}")
        return False


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
    "--all-layers",
    "-al",
    is_flag=True,
    help="If present, set feature extraction to False and train all layers, train only the last layer otherwise.",
)
@click.option(
    "--from-scratch",
    "-fs",
    is_flag=True,
    help="If present, trains the networks from scratch without using pretrained weights.",
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
@click.option(
    "--run_name", "-rn", type=str, default="default", help="Base name for the wandb run"
)
@click.option(
    "--upload_to_hf", is_flag=True, help="Upload best model to HuggingFace Hub"
)
@click.option(
    "--hf_repo_id",
    type=str,
    default="",
    help="HuggingFace repository ID for uploading (e.g., dushj98/model-name)",
)
def train(
    dataset_name: str,
    hf_token: str,
    wandb_token: str,
    epochs: int,
    lr: float,
    all_layers: bool,
    from_scratch: bool,
    metadata: str,
    models: str,
    seed: int,
    run_name: str,
    upload_to_hf: bool,
    hf_repo_id: str,
) -> None:
    # Correctly set the training mode
    if from_scratch:
        print("⚠ Training from scratch: overriding --all-layers to True")
        all_layers = True
        training_mode = "pt"

    else:
        training_mode = "ft" if all_layers else "tl"

    feature_extract = not all_layers
    if from_scratch and feature_extract:
        raise ValueError(
            "Feature extraction cannot be enabled when training from scratch."
        )

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
    try:
        import wandb

        wandb.login(key=wandb_token)
        print("Successfully logged in to Weights & Biases")
    except Exception as e:
        print(f"Failed to login to Weights & Biases: {e}")
        return

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

    # Add reproducibility info, feature extraction,
    # and training mode to metadata
    metadata_dict["random_seed"] = seed
    metadata_dict["feature_extraction"] = feature_extract
    metadata_dict["training_mode"] = training_mode

    # Check HF upload options
    if upload_to_hf and not hf_repo_id:
        print("Warning: --upload_to_hf flag provided but no --hf_repo_id specified.")
        print("Will generate a default repo ID based on model name and dataset.")

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
    print(f"💻 Training mode: {training_mode_map[training_mode]}")

    num_classes = 13

    # Train each model sequentially
    all_results = {}
    for model_name in models_to_train:
        print(f"\n{'=' * 50}")
        print(f"Training {model_name} on {dataset_name}")
        print(f"{'=' * 50}")

        # End any existing W&B run
        if wandb.run is not None:
            wandb.finish()

        # Initialize W&B run with config for this model
        config = {
            "model": model_name,
            "epochs": epochs,
            "lr": lr,
            "batch_size": model_configs[model_name]["batch_size"],
            "dataset": dataset_name,
            **metadata_dict,
        }

        wandb_run_name = f"{model_name}_{run_name}_{'tl' if feature_extract else 'ft'}"

        try:
            wandb.init(
                project=f"aerial-classification-{run_name.replace('_', '-')}",
                name=wandb_run_name,
                config=config,
            )
            print(f"W&B run initialized for {model_name} with config: {config}")
        except Exception as e:
            print(f"Failed to initialize W&B run: {e}")
            return

        model_ft, input_size = initialize_model(
            model_name,
            num_classes,
            feature_extract=feature_extract,
            use_pretrained=not from_scratch,
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
        trained_model, results, checkpoint_path = train_model(
            model_ft,
            dataloaders_dict,
            criterion,
            optimizer,
            num_epochs=epochs,
            model_name=f"{model_name}_{run_name}_{training_mode}",
        )

        # Store results
        all_results[model_name] = results

        # Upload to HuggingFace if requested
        if upload_to_hf:
            # Generate repo ID if not provided
            model_repo_id = hf_repo_id
            if not model_repo_id:
                # Generate a default name based on model
                model_repo_id = f"dushj98/aerial-{model_name.replace('_', '-')}-{run_name.replace('_', '-')}-{training_mode}"
                print(f"Using generated repository ID: {model_repo_id}")

            print(f"Uploading {model_name} to HuggingFace Hub repo {model_repo_id}...")
            upload_success = upload_model_to_hf(
                model_name=model_name,
                dataset_name=dataset_name,
                hf_repo_id=model_repo_id,
                checkpoint_path=checkpoint_path,
                num_classes=num_classes,
                metadata={**config, **results},
            )

            if upload_success:
                print(f"Model uploaded successfully to {model_repo_id}")
            else:
                print(f"Failed to upload model to HuggingFace")

        # Make sure wandb is closed
        if wandb.run is not None:
            wandb.finish()

    # Print summary of results
    print("\n" + "=" * 50)
    print("TRAINING SUMMARY")
    print("=" * 50)
    print(f"Dataset: {dataset_name}")
    print(f"Random Seed: {seed}")  # Print the seed used
    print(
        f"Training Mode: {'Feature extraction' if feature_extract else 'Fine-tuning'}"
    )

    header = f"{'Model':<15} | {'Best Val Acc':<10} | {'Best Epoch':<10} | {'Train Time':<10}"
    print("\n" + header)
    print("-" * len(header))

    for model_name, results in all_results.items():
        acc = results["best_val_acc"] * 100
        epoch = results["best_epoch"]
        time_mins = results["train_time"] / 60
        print(f"{model_name:<15} | {acc:>9.2f}% | {epoch:>10d} | {time_mins:>9.1f}m")

    # Save results with seed for reproducibility tracking
    results_filename = f"training_results_{run_name.replace('-', '_')}_seed{seed}_{'tl' if feature_extract else 'ft'}.txt"
    try:
        with open(results_filename, "w") as f:
            f.write(f"Dataset: {dataset_name}\n")
            f.write(f"Random Seed: {seed}\n")
            f.write(
                f"Training Mode: {'Feature extraction' if feature_extract else 'Fine-tuning'}\n\n"
            )
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
