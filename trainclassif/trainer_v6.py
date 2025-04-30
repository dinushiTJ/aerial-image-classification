import copy
import json
import os
import random
import time
import traceback
from typing import Any

import click
import numpy as np
import torch
import wandb
from datasets import get_dataset_infos
from datasets import load_dataset
from sklearn.metrics import f1_score
from sklearn.metrics import precision_score
from sklearn.metrics import recall_score
from tabulate import tabulate
from torch import device
from torch import set_grad_enabled
from torch.cuda import is_available as is_cuda_available
from torch.nn import CrossEntropyLoss
from torch.nn import Dropout
from torch.nn import Linear
from torch.nn import Module
from torch.nn import Sequential
from torch.optim import Adam
from torch.optim import AdamW
from torch.optim import Optimizer
from torch.optim.lr_scheduler import CosineAnnealingLR
from torch.optim.lr_scheduler import LRScheduler
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import DataLoader
from torchvision import models
from torchvision import transforms
from tqdm.auto import tqdm

HF_TOKEN_ENV_VAR = "HF_TOKEN"
HF_USER_ENV_VAR = "HF_USER"
WANDB_API_KEY_ENV_VAR = "WANDB_API_KEY"

training_mode_map = {
    "tl": "Transfer Learning (Head Only)",
    "sft": "Selective Fine-Tuning",
    "fft": "Full Fine-Tuning",
}

# Model configurations including separate learning rates for TL and FT/SFT/PT
model_configs = {
    "mobilenet_v2": {
        "batch_size": 64,
        "input_size": 224,
        "base_lr_tl": 5e-4,
        "base_lr_ft": 2e-5,
        "unfreeze": [
            "features.14.",
            "features.15.",
            "features.16.",
            "features.17.",
            "features.18.",
        ],
    },
    "resnet50": {
        "batch_size": 32,
        "input_size": 224,
        "base_lr_tl": 3e-4,
        "base_lr_ft": 1e-5,
        "unfreeze": ["layer4."],
    },
    "efficientnet_b2": {
        "batch_size": 32,
        "input_size": 224,
        "base_lr_tl": 3e-4,
        "base_lr_ft": 1e-5,
        "unfreeze": ["features.6.", "features.7.", "features.8."],
    },
    "efficientnetv2_m": {
        "batch_size": 16,
        "input_size": 224,
        "base_lr_tl": 2e-4,
        "base_lr_ft": 1e-5,
        "unfreeze": ["features.6", "features.7", "features.8"],
    },
    "vit_b_16": {
        "batch_size": 16,
        "input_size": 224,
        "base_lr_tl": 1e-4,
        "base_lr_ft": 1e-5,
        "unfreeze": [
            "encoder.layers.encoder_layer_8.",
            "encoder.layers.encoder_layer_9.",
            "encoder.layers.encoder_layer_10.",
            "encoder.layers.encoder_layer_11.",
        ],
    },
}


def set_reproducibility(seed: int = 42) -> callable:
    """
    Sets random seeds and CUDA flags for reproducibility.

    Args:
        seed (int): The seed value.

    Returns:
        A worker_init_fn for DataLoader reproducibility.
    """
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if is_cuda_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)  # For multi-GPU
        # Deterministic operations can impact performance and might not cover all ops
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False  # Disable benchmark for determinism

    def seed_worker(worker_id: int):
        """Worker init function for DataLoader."""
        worker_seed = torch.initial_seed() % 2**32  # Get dataloader base seed
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    print(f"🌱 Reproducibility seed set to {seed}")
    return seed_worker


def get_dataset_stats(
    dataset_name: str, 
    split: str, 
    n_samples: int, 
    input_size: int
) -> tuple[list | None, list | None]:
    """
    Calculates approximate mean and std for a dataset split using streaming.

    Args:
        dataset_name: Name of the Hugging Face dataset.
        split: Dataset split to use (e.g., 'train').
        n_samples: Number of samples to use for calculation.
        input_size: The target input size for resizing/cropping.

    Returns:
        A tuple containing the list of means and list of stds, or (None, None) on failure.
    """
    print(
        f"⏳ Calculating approximate dataset stats for '{dataset_name}' [{split}] split using {n_samples} samples..."
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    count = 0

    try:
        # Use streaming to avoid downloading the full dataset upfront
        dataset = load_dataset(dataset_name, split=split, streaming=True)
        # Shuffle and take a sample for calculation
        sampled_dataset = dataset.shuffle(seed=42, buffer_size=n_samples * 5).take(
            n_samples
        )

        # Basic transform to tensor for calculation
        # Resize slightly larger than crop size before CenterCrop
        transform = transforms.Compose(
            [
                transforms.Resize(int(input_size * 1.15)),  # Resize slightly larger
                transforms.CenterCrop(input_size),
                transforms.ToTensor(),
            ]
        )

        # Iterate through the sampled dataset
        for example in tqdm(
            sampled_dataset, total=n_samples, desc="Calculating Stats", leave=False
        ):
            try:
                img = example["image"]
                # Ensure image is PIL and RGB
                if not hasattr(img, "convert"):
                    # If not PIL, try to convert (might fail for some formats)
                    img = transforms.ToPILImage()(img)
                if img.mode != "RGB":
                    img = img.convert("RGB")

                img_tensor = transform(img)  # Shape: C x H x W
                # Calculate mean and std per channel across H and W dimensions
                mean += img_tensor.mean(dim=[1, 2])
                std += img_tensor.std(dim=[1, 2])
                count += 1

            except Exception as e:
                print(
                    f"⚠️ Warning: Skipping image during stats calculation due to error: {e}"
                )
                continue

        if count == 0:
            print("🚨 Error: No valid images processed for stats calculation.")
            return None, None

        # Calculate final mean and std
        mean /= count
        std /= count
        mean_list = mean.tolist()
        std_list = std.tolist()
        print(f"📊 Calculated Mean: {[f'{x:.3f}' for x in mean_list]}")
        print(f"📊 Calculated Std:  {[f'{x:.3f}' for x in std_list]}")
        return mean_list, std_list

    except Exception as e:
        print(f"🚨 Failed to calculate dataset stats: {e}")
        print(traceback.format_exc())
        return None, None  # Fallback indicates failure


def load_data(
    dataset_name: str,
    input_size: int,
    batch_size: int,
    seed_worker: callable,
    g: torch.Generator,
    use_augmentation: bool = False,
    dataset_mean: list | None = None,
    dataset_std: list | None = None,
) -> dict[str, DataLoader]:
    """
    Loads and preprocesses the dataset with optional augmentation and custom normalization.

    Args:
        dataset_name: Name of the Hugging Face dataset.
        input_size: Target input size for the model.
        batch_size: Number of samples per batch.
        seed_worker: Worker init function for DataLoader reproducibility.
        g: PyTorch random generator for DataLoader shuffling.
        use_augmentation: Whether to apply enhanced data augmentation to the training set.
        dataset_mean: Optional list of means for normalization. Uses ImageNet if None.
        dataset_std: Optional list of standard deviations for normalization. Uses ImageNet if None.

    Returns:
        A dictionary containing 'train' and 'val' DataLoaders.
    """
    print(f"🔃 Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name)

    # Determine normalization transform
    if dataset_mean and dataset_std:
        print(
            f"   Using dataset-specific normalization: mean={dataset_mean}, std={dataset_std}"
        )
        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
    else:
        print("   Using ImageNet default normalization.")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    # Define transformations
    # Resize slightly larger than input size for cropping
    resize_size = int(input_size * 1.143)  # e.g., 256 for 224 input

    # Validation transform (minimal augmentation)
    val_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    # Training transform
    if use_augmentation:
        print("   Applying ENHANCED data augmentation to training set.")
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(input_size, scale=(0.7, 1.0)),
                transforms.RandomAffine(degrees=15, translate=(0.1, 0.1), shear=10),
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),
                transforms.RandomRotation(15),
                transforms.ColorJitter(
                    brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1
                ),
                transforms.ToTensor(),
                normalize,
                transforms.GaussianBlur(kernel_size=(5, 9), sigma=(0.1, 5.0)),
                transforms.RandomErasing(p=0.25, scale=(0.02, 0.2)),
            ]
        )
    else:
        print("   Applying BASIC transformations to training set.")
        train_transform = val_transform

    def _ensure_rgb(image):
        """Converts image to RGB if not already."""
        if not hasattr(image, "convert"):
            try:
                image = transforms.ToPILImage()(image)
            except Exception as e:
                raise TypeError(
                    f"Input image could not be converted to PIL. Type: {type(image)}. Error: {e}"
                )
        if image.mode != "RGB":
            return image.convert("RGB")
        return image

    def transform_dataset(example: dict, split: str) -> dict:
        """Applies the appropriate transform to a single example."""
        try:
            image = _ensure_rgb(example["image"])
            transform_to_apply = train_transform if split == "train" else val_transform
            example["image"] = transform_to_apply(image)
            return example
        except Exception as e:
            print(
                f"🚨 Error processing image in {split} set: {e}. Example keys: {list(example.keys())}"
            )
            # Depending on dataset, might need to return None or skip
            raise  # Re-raise error to potentially stop processing if critical

    # Apply transformations using .map()
    print("   Applying transformations...")
    dataset["train"] = dataset["train"].map(
        lambda ex: transform_dataset(ex, "train"),
        batched=False,  # Process image by image
        desc="Applying train transforms",
        load_from_cache_file=False,
    )
    dataset["validation"] = dataset["validation"].map(
        lambda ex: transform_dataset(ex, "val"),
        batched=False,
        desc="Applying validation transforms",
        load_from_cache_file=False,
    )

    # Set format for PyTorch
    dataset.set_format(type="torch", columns=["image", "label"])

    # Create DataLoaders
    num_workers = os.cpu_count() // 2 if os.cpu_count() else 4
    print(f"   Using {num_workers} workers for DataLoaders.")
    dataloaders = {
        "train": DataLoader(
            dataset["train"],
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=is_cuda_available(),  # Pin memory if using GPU
            persistent_workers=(
                True if num_workers > 0 and is_cuda_available() else False
            ),
            drop_last=True,
        ),
        "val": DataLoader(
            dataset["validation"],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            worker_init_fn=seed_worker,
            generator=g,
            pin_memory=is_cuda_available(),
            persistent_workers=(
                True if num_workers > 0 and is_cuda_available() else False
            ),
        ),
    }
    print("✅ DataLoaders created.")
    return dataloaders


def set_parameter_requires_grad(model: Module, requires_grad: bool) -> None:
    """Sets requires_grad for all model parameters."""
    for param in model.parameters():
        param.requires_grad = requires_grad


def modify_last_layer(
    model_name: str, model: Module, num_classes: int, dropout_p: float
) -> None:
    if dropout_p < 0 or dropout_p > 1:
        raise ValueError("Dropout value must be a valid percentage.")

    if dropout_p > 0:
        if model_name in ["mobilenet_v2", "efficientnet_b2", "efficientnetv2_m"]:
            num_ftrs = model.classifier[1].in_features
            model.classifier = Sequential(
                Dropout(p=dropout_p, inplace=True),  # inplace=True saves memory
                Linear(num_ftrs, num_classes),
            )

        elif model_name == "resnet50":
            num_ftrs = model.fc.in_features
            model.fc = Sequential(
                Dropout(p=dropout_p, inplace=True), Linear(num_ftrs, num_classes)
            )

        elif model_name == "vit_b_16":
            # ViT head structure might vary slightly depending on torchvision version
            if hasattr(model, "heads") and isinstance(model.heads, Sequential):
                num_ftrs = model.heads[-1].in_features
                model.heads[-1] = Sequential(
                    Dropout(p=dropout_p, inplace=True), Linear(num_ftrs, num_classes)
                )

            elif hasattr(model, "heads") and hasattr(model.heads, "head"):
                # Older structure with heads.head
                num_ftrs = model.heads.head.in_features
                model.heads.head = Sequential(
                    Dropout(p=dropout_p, inplace=True), Linear(num_ftrs, num_classes)
                )

            else:
                raise AttributeError(
                    f"Cannot determine classifier head structure for ViT model: {model_name}"
                )

    else:
        if model_name in ["mobilenet_v2", "efficientnet_b2", "efficientnetv2_m"]:
            num_ftrs = model.classifier[1].in_features
            model.classifier[1] = Linear(num_ftrs, num_classes)

        elif model_name == "resnet50":
            num_ftrs = model.fc.in_features
            model.fc = Linear(num_ftrs, num_classes)

        elif model_name == "vit_b_16":
            num_ftrs = model.heads.head.in_features
            model.heads.head = Linear(num_ftrs, num_classes)

    classifier_module = None
    if model_name in ["mobilenet_v2", "efficientnet_b2", "efficientnetv2_m"]:
        classifier_module = model.classifier

    elif model_name == "resnet50":
        classifier_module = model.fc

    elif model_name == "vit_b_16":
        if hasattr(model, "heads") and isinstance(model.heads, Sequential):
            classifier_module = model.heads[-1]

        elif hasattr(model, "heads") and hasattr(model.heads, "head"):
            classifier_module = model.heads.head

    if classifier_module:
        for name, param in classifier_module.named_parameters():
            if not param.requires_grad:
                print(
                    f"   🚨 WARNING: Classifier parameter '{name}' is NOT trainable! Forcing requires_grad=True."
                )
                param.requires_grad = True
    else:
        raise ValueError(
            "⚠️ Could not definitively verify classifier trainability due to unknown structure."
        )


def initialize_model(
    model_name: str,
    num_classes: int,
    requires_grad: bool,
    dropout_p: float,
    layers_to_unfreeze: list[str],
) -> Module:
    """
    Initializes the model, sets gradient requirements based on training mode.

    Args:
        model_name: Name of the model architecture (must be in model_configs).
        num_classes: Number of output classes.
        feature_extract: If True, freeze base model and train only the head initially.
                         If False, all layers start as trainable (for FT/PT).
        layers_to_unfreeze: list of named parameter/module keywords to unfreeze (used in 'sft' mode).
                            Applied *after* initial feature_extract freezing.
        use_pretrained: Whether to use ImageNet pretrained weights.
        dropout_p: Probability for the dropout layer before the classifier.

    Returns:
        Initialized model and its input size.
    """
    print(f"🛠️ Initializing model: {model_name}")
    if model_name not in model_configs:
        raise ValueError(f"Model '{model_name}' not found in model_configs.")

    model = None

    if model_name == "mobilenet_v2":
        model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V2)
        set_parameter_requires_grad(model=model, requires_grad=requires_grad)
        modify_last_layer(
            model_name=model_name,
            model=model,
            num_classes=num_classes,
            dropout_p=dropout_p,
        )

    elif model_name == "resnet50":
        model = models.resnet50(weights=models.ResNet50_Weights.IMAGENET1K_V2)
        set_parameter_requires_grad(model=model, requires_grad=requires_grad)
        modify_last_layer(
            model_name=model_name,
            model=model,
            num_classes=num_classes,
            dropout_p=dropout_p,
        )

    elif model_name == "efficientnet_b2":
        model = models.efficientnet_b2(
            weights=models.EfficientNet_B2_Weights.IMAGENET1K_V1
        )
        set_parameter_requires_grad(model=model, requires_grad=requires_grad)
        modify_last_layer(
            model_name=model_name,
            model=model,
            num_classes=num_classes,
            dropout_p=dropout_p,
        )

    elif model_name == "efficientnetv2_m":
        model = models.efficientnet_v2_m(
            weights=models.EfficientNet_V2_M_Weights.IMAGENET1K_V1
        )
        set_parameter_requires_grad(model=model, requires_grad=requires_grad)
        modify_last_layer(
            model_name=model_name,
            model=model,
            num_classes=num_classes,
            dropout_p=dropout_p,
        )

    elif model_name == "vit_b_16":
        model = models.vit_b_16(weights=models.ViT_B_16_Weights.IMAGENET1K_V1)
        set_parameter_requires_grad(model=model, requires_grad=requires_grad)
        modify_last_layer(
            model_name=model_name,
            model=model,
            num_classes=num_classes,
            dropout_p=dropout_p,
        )

    # Selective Unfreezing (if layers_to_unfreeze provided)
    if layers_to_unfreeze:
        print(
            f"   Selectively unfreezing layers containing keywords: {layers_to_unfreeze}"
        )
        unfrozen_count = 0

        for name, param in model.named_parameters():
            if any(layer_key in name for layer_key in layers_to_unfreeze):
                if not param.requires_grad:
                    param.requires_grad = True
                unfrozen_count += 1

        print(f"   Unfroze {unfrozen_count} parameters matching keywords.")

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    print(f"   Model: {model_name}")
    print(
        f"   Trainable Parameters: {num_trainable:,} / {num_total:,} ({num_trainable/num_total:.2%})"
    )

    return model


def train_model(
    model: Module,
    model_name: str,
    dataloaders: dict[str, DataLoader],
    criterion: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler,
    num_epochs: int,
    patience: int,
    use_mixed_precision: bool,
) -> tuple[Module, dict[str, Any], str | None]:  # Return model, results dict, checkpoint path
    """
    Trains the model for a specified number of epochs with early stopping.

    Args:
        model: The PyTorch model to train.
        dataloaders: dictionary of DataLoaders for 'train' and 'val'.
        criterion: The loss function.
        optimizer: The optimizer.
        scheduler: Optional learning rate scheduler.
        num_epochs: Maximum number of epochs to train.
        patience: Number of epochs to wait for validation accuracy improvement before stopping.
        model_name: Base name for saving the best checkpoint file.
        use_mixed_precision: Whether to use Automatic Mixed Precision (AMP).

    Returns:
        A tuple containing:
        - The model with the best validation accuracy weights loaded.
        - A dictionary containing training history and results.
        - The path to the saved best model checkpoint (or None if no improvement).
    """
    start_time = time.time()
    val_acc_history = []
    best_epoch = -1
    best_acc = 0.0
    best_f1 = 0.0
    best_loss = float("inf")
    best_model_wts = copy.deepcopy(model.state_dict())  # Store initial weights
    best_checkpoint_path = None  # Path to the best saved model
    results: dict[str, Any] = {
        "best_val_acc": 0.0,
        "best_val_loss": float("inf"),
        "best_epoch": -1,
        "val_acc_history": [],
        "val_loss_history": [],
        "train_acc_history": [],
        "train_loss_history": [],
        "lr_history": [],
        "train_time": 0.0,
    }  # Initialize results dict

    dev = device("cuda:0" if is_cuda_available() else "cpu")
    model.to(dev)
    print(f"🏋️‍♀️ Starting training on device: {dev}")

    # Initialize Gradient Scaler for Mixed Precision if enabled and available
    scaler = None
    amp_enabled = use_mixed_precision and dev.type == "cuda"
    if amp_enabled:
        try:
            from torch.amp import GradScaler
            from torch.amp import autocast

            scaler = GradScaler()
            print("   ⚡ Automatic Mixed Precision (AMP) Enabled.")
        except ImportError:
            print("   ⚠️ Warning: torch.amp not available. Disabling Mixed Precision.")
            amp_enabled = False  # Disable if import fails
    else:
        print(f"   Automatic Mixed Precision (AMP) {'Disabled' if not use_mixed_precision else 'Unavailable (not on CUDA)'}.")

    # Epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0, leave=True)
    epochs_no_improve = 0  # Counter for early stopping

    for epoch in epoch_pbar:
        epoch_metrics: dict[str, float] = {}  # Store metrics for this epoch
        current_lr = optimizer.param_groups[0]["lr"]  # Get current LR
        results["lr_history"].append(current_lr)

        for phase in ["train", "val"]:
            is_train = (phase == "train")
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # Batch progress bar
            batch_pbar = tqdm(
                dataloaders[phase],
                desc=f"{phase.capitalize():<5}",  # e.g., "Train", "Val  "
                position=1,  # Inner bar
                leave=False,  # Remove after completion
                total=len(dataloaders[phase]),
            )

            for batch in batch_pbar:
                inputs = batch["image"].to(dev, non_blocking=True)
                labels = batch["label"].to(dev, non_blocking=True)

                # Zero gradients before forward pass
                optimizer.zero_grad(set_to_none=True)  # More memory efficient

                # Forward pass with autocasting for AMP if enabled
                with set_grad_enabled(is_train):  # Enable grads only in training
                    if amp_enabled:
                        from torch.amp import autocast  # Import locally for clarity

                        with autocast(device_type="cuda:0" if is_cuda_available() else "cpu"):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:  # Default precision
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # Get predictions (index of max logit)
                    _, preds = torch.max(outputs, 1)

                # Backward pass + optimize only if in training phase
                if is_train:
                    if scaler:  # Using AMP scaler
                        scaler.scale(loss).backward()  # Scale loss
                        # Optional: Gradient Clipping (unscale first)
                        # scaler.unscale_(optimizer)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)  # Optimizer step (unscales internally)
                        scaler.update()  # Update scaler for next iteration
                    else:  # Default precision
                        loss.backward()
                        # Optional: Gradient Clipping
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                # Statistics update
                batch_loss = loss.item()
                running_loss += batch_loss * inputs.size(0)  # Accumulate loss scaled by batch size
                running_corrects += torch.sum(preds == labels.data).item()  # Accumulate correct predictions

                # Store predictions and labels for epoch-level metrics (F1, etc.)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # Update batch progress bar postfix
                batch_pbar.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    lr=f"{current_lr:.1E}",
                )  # Show batch loss and LR

            # Calculate epoch metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            epoch_metrics[f"{phase}_loss"] = epoch_loss
            epoch_metrics[f"{phase}_acc"] = epoch_acc
            results[f"{phase}_loss_history"].append(epoch_loss)
            results[f"{phase}_acc_history"].append(epoch_acc)

            # Calculate Precision, Recall, F1 (macro average)
            # zero_division=0 handles cases where a class might not be predicted in a batch/epoch
            epoch_precision = precision_score(all_labels, all_preds, average="macro", zero_division=0)
            epoch_recall = recall_score(all_labels, all_preds, average="macro", zero_division=0)
            epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            epoch_metrics[f"{phase}_precision"] = epoch_precision
            epoch_metrics[f"{phase}_recall"] = epoch_recall
            epoch_metrics[f"{phase}_f1"] = epoch_f1

            # Validation phase specific actions
            if phase == "val":
                val_acc_history.append(epoch_acc)  # Keep track for early stopping check

                # Step the scheduler based on validation metric (if applicable)
                if scheduler is not None:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(epoch_acc)  # Step based on validation accuracy
                    
                    else:
                        scheduler.step()  # CosineAnnealingLR steps every epoch regardless of metrics

                # Check for improvement & early stopping
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss  # Also track loss at best accuracy
                    best_epoch = epoch
                    best_f1 = epoch_f1
                    best_model_wts = copy.deepcopy(model.state_dict())  # Save best weights
                    epochs_no_improve = 0  # Reset counter

                    # Save the best model checkpoint
                    best_checkpoint_path = f"best_checkpoint_{model_name}.pt"
                    try:
                        torch.save(model.state_dict(), best_checkpoint_path)
                        tqdm.write(f"✅ Epoch {epoch}: Val Acc improved to {best_acc:.4f}. Checkpoint saved to '{best_checkpoint_path}'")  # Use tqdm.write to avoid interfering with progress bars
                        # Save to WandB safely
                        if wandb.run:
                            try:
                                artifact = wandb.Artifact(f"{model_name}-best-model", type="model")
                                artifact.add_file(best_checkpoint_path)
                                wandb.log_artifact(artifact)
                                tqdm.write(f"   🦴 Checkpoint artifact saved to WandB.")
                            except Exception as wb_err:
                                tqdm.write(f"   ⚠️ Failed to save checkpoint artifact to WandB: {wb_err}")

                    except Exception as e:
                        tqdm.write(f"   🚨 Failed to save checkpoint: {e}")
                        best_checkpoint_path = None  # Reset path if saving failed
                else:
                    epochs_no_improve += 1
                    tqdm.write(f"📉 Epoch {epoch}: Val Acc ({epoch_acc:.4f}) did not improve from best ({best_acc:.4f}). ({epochs_no_improve}/{patience})")

        # Log epoch metrics to WandB safely
        if wandb.run:
            log_data = {
                f"{m}/{p}": epoch_metrics.get(f"{p}_{m}", 0.0)
                for p in ["train", "val"]
                for m in ["loss", "acc", "precision", "recall", "f1"]
            }
            log_data["epoch"] = epoch
            log_data["learning_rate"] = current_lr
            log_data["epochs_since_improvement"] = epochs_no_improve
            try:
                wandb.log(log_data)
            except Exception as e:
                tqdm.write(f"   🚨 WandB logging failed for epoch {epoch}: {e}")

        # Update epoch progress bar postfix with key metrics
        epoch_pbar.set_postfix(
            TrL=f"{epoch_metrics.get('train_loss', 0):.3f}",
            TrA=f"{epoch_metrics.get('train_acc', 0):.3f}",
            VL=f"{epoch_metrics.get('val_loss', 0):.3f}",
            VA=f"{epoch_metrics.get('val_acc', 0):.3f}",
            BestVA=f"{best_acc:.3f}",
            LR=f"{current_lr:.1E}",
        )

        # Print epoch summary to console
        tqdm.write(
            f"Epoch {epoch}/{num_epochs-1} -> "
            f"Train[L:{epoch_metrics.get('train_loss', 0):.4f} A:{epoch_metrics.get('train_acc', 0):.4f} F1:{epoch_metrics.get('train_f1', 0):.4f}] | "
            f"Val[L:{epoch_metrics.get('val_loss', 0):.4f} A:{epoch_metrics.get('val_acc', 0):.4f} F1:{epoch_metrics.get('val_f1', 0):.4f}] | "
            f"LR: {current_lr:.6f}"
        )  # Detailed epoch summary

        # Early stopping check
        if epochs_no_improve >= patience:
            tqdm.write(f"\n⏳ Early stopping triggered at epoch {epoch} after {patience} epochs with no improvement.")
            break  # Exit epoch loop

    end_time = time.time()
    total_time = end_time - start_time
    results["train_time"] = total_time
    print(f"\n🏁 Training finished in {total_time // 60:.0f}m {total_time % 60:.1f}s")

    # Load best model weights if an improvement was found
    if best_epoch != -1:
        print(f"🏆 Best Validation Acc: {best_acc:.4f} achieved at Epoch {best_epoch}.")
        print(f"   Loading weights from: {best_checkpoint_path}")
        try:
            model.load_state_dict(torch.load(best_checkpoint_path))
            results["best_val_acc"] = best_acc
            results["best_val_loss"] = best_loss  # Loss corresponding to best accuracy
            results["best_val_f1"] = best_f1  # f1 corresponding to best accuracy
            results["best_epoch"] = best_epoch
        except Exception as e:
            print(f"🚨 Error loading best model weights from {best_checkpoint_path}: {e}")
            best_checkpoint_path = None  # Invalidate path if loading fails
    else:
        print("⚠️ No improvement in validation accuracy was observed during training.")
        # Keep initial weights, results reflect last epoch or initial state
        results["best_val_acc"] = val_acc_history[0] if val_acc_history else 0.0
        results["best_val_loss"] = (
            results["val_loss_history"][0]
            if results["val_loss_history"]
            else float("inf")
        )
        results["best_val_f1"] = 0.0
        results["best_epoch"] = 0

    results["val_acc_history"] = val_acc_history  # Store full history

    # Return the model (with best weights loaded if possible), results, and best checkpoint path
    return model, results, best_checkpoint_path


def get_hf_config(
    model_name: str, num_classes: int, id2label: dict, label2id: dict
) -> dict:
    """Generates a basic Hugging Face model config.json."""
    return {
        "_name_or_path": model_name,  # Informational
        "architectures": [
            f"{model_name.capitalize()}ForImageClassification"
        ],  # Placeholder, adjust if needed
        "model_type": model_name.split("_")[0],  # e.g., 'resnet', 'vit'
        "num_labels": num_classes,
        "id2label": id2label,
        "label2id": label2id,
        "problem_type": "single_label_classification",
    }


def get_hf_preprocessor_config(
    model_name: str,
    input_size: int,
    dataset_mean: list | None,
    dataset_std: list | None,
) -> dict:
    """Generates a basic Hugging Face preprocessor_config.json."""
    # Use dataset stats if provided, otherwise ImageNet defaults
    effective_mean = dataset_mean if dataset_mean else [0.485, 0.456, 0.406]
    effective_std = dataset_std if dataset_std else [0.229, 0.224, 0.225]
    resize_size = int(input_size * 1.143)

    return {
        "do_normalize": True,
        "do_rescale": False,  # Normalization handles rescaling if ToTensor used
        "do_resize": True,
        "image_mean": effective_mean,
        "image_std": effective_std,
        "resample": transforms.InterpolationMode.BILINEAR.value,  # Standard resize interpolation
        "size": {"shortest_edge": resize_size},  # Standard HF resize convention
        "crop_size": {"height": input_size, "width": input_size},  # Crop size used
        # Choose a common processor type, may need adjustment based on model
        "image_processor_type": "ConvNextImageProcessor",
    }


def create_hf_model_card(
    model_name: str,
    dataset_name: str,
    num_classes: int,
    id2label: dict,
    metadata: dict,
    dataset_mean: list | None,
    dataset_std: list | None,
) -> str:
    """Creates a Markdown model card string."""
    lr = metadata.get("target_lr", "N/A")
    lr_str = f"{lr:.1E}" if isinstance(lr, float) else "N/A"
    best_acc = metadata.get("best_val_acc", 0.0)
    best_f1 = metadata.get("best_val_f1", 0.0)

    card = f"""
---
language: en
library_name: pytorch
tags:
- image-classification
- pytorch
- {model_name}
- {dataset_name.replace("/", "-")}
- aerial-imagery
- generated-by-trainer-script
datasets:
- {dataset_name}
metrics:
- accuracy
- f1
---

# {model_name} fine-tuned on {dataset_name}

This model is a version of `{model_name}` fine-tuned on the `{dataset_name}` dataset for aerial image classification.

## Model Details

- **Model Architecture:** `{model_name}`
- **Pretrained Weights:** {'ImageNet (Default)'}
- **Training Mode:** {training_mode_map.get(metadata.get('training_mode', 'N/A'), 'N/A')}
- **Number of Classes:** {num_classes}
- **Input Size:** {metadata.get('input_size', 'N/A')}x{metadata.get('input_size', 'N/A')}
- **Labels:** {', '.join(id2label.values())}

## Training Configuration

- **Dataset:** `{dataset_name}`
- **Optimizer:** {metadata.get('optimizer', 'N/A')}
- **Learning Rate (Initial):** {lr_str}
- **Scheduler:** {metadata.get('scheduler', 'None')}
- **Epochs:** {metadata.get('epochs', 'N/A')} (Target), {metadata.get('best_epoch', 'N/A')} (Best Epoch)
- **Batch Size:** {metadata.get('batch_size', 'N/A')}
- **Label Smoothing:** {metadata.get('label_smoothing', 0.0)}
- **Dropout:** {metadata.get('dropout_p', 0.5)}
- **Mixed Precision:** {metadata.get('use_mixed_precision', False)}
- **Data Augmentation:** {'Yes (Enhanced)' if metadata.get('data_augmentation', False) else 'No (Basic)'}
- **Normalization Mean:** {f'{dataset_mean}' if dataset_mean else '[ImageNet Default]'}
- **Normalization Std:** {f'{dataset_std}' if dataset_std else '[ImageNet Default]'}
- **Seed:** {metadata.get('seed', 'N/A')}

## Performance

- **Best Validation Accuracy:** {best_acc:.4f}
- **Best Validation F1-Score:** {best_f1:.4f}
- **Training Time:** {metadata.get('train_time', 0.0) / 60:.1f} minutes

*Include plots or tables from training logs (e.g., WandB) if desired.*

## How to Use (`transformers`)

```python
from transformers import AutoImageProcessor, AutoModelForImageClassification
import torch
from PIL import Image
import requests

# Define repository ID and load processor/model
repo_id = "{{hf_repo_id}}"
processor = AutoImageProcessor.from_pretrained(repo_id)
model = AutoModelForImageClassification.from_pretrained(repo_id)

# Example image URL (replace with your image)
# url = "[https://example.com/your_aerial_image.jpg](https://example.com/your_aerial_image.jpg)"
# image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
# Or load from file:
# image = Image.open("path/to/your/image.jpg").convert("RGB")

# --- Placeholder: Load a sample image ---
try:
    url = "[http://images.cocodataset.org/val2017/000000039769.jpg](http://images.cocodataset.org/val2017/000000039769.jpg)" # Example COCO image
    image = Image.open(requests.get(url, stream=True).raw).convert("RGB")
    print("Loaded sample image.")
except Exception as e:
    print(f"Could not load sample image: {{e}}. Please provide your own image.")
    image = Image.new('RGB', (224, 224), color = 'red') # Dummy image

# Preprocess image
inputs = processor(images=image, return_tensors="pt")

# Make prediction
with torch.no_grad():
    logits = model(**inputs).logits

# Get predicted class index and label
predicted_label_idx = logits.argmax(-1).item()
predicted_label = model.config.id2label[predicted_label_idx]

print(f"Predicted class: {{predicted_label}} (ID: {{predicted_label_idx}})")
```

## Intended Use & Limitations

This model is intended for classifying aerial images based on the categories present in the `{dataset_name}` dataset. Performance may vary on images significantly different from the training distribution. Evaluate carefully for your specific use case. The model inherits limitations from the base `{model_name}` architecture and ImageNet pretraining (if used).
"""
    return card.strip()


def upload_model_to_hf(
    model_name: str,
    dataset_name: str,
    hf_repo_id: str,
    checkpoint_path: str,
    num_classes: int,
    id2label: dict,
    label2id: dict,
    metadata: dict,
    dataset_mean: list | None,
    dataset_std: list | None,
) -> bool:
    """
    Uploads the trained model, config, preprocessor config, and model card to the Hugging Face Hub.

    Args:
        model_name: Name of the model architecture.
        dataset_name: Name of the training dataset.
        hf_repo_id: Target Hugging Face repository ID (e.g., "username/repo-name").
        checkpoint_path: Local path to the best model checkpoint (.pt file).
        num_classes: Number of classes.
        id2label: dictionary mapping class index to label name.
        label2id: dictionary mapping label name to class index.
        metadata: dictionary containing training configuration and results.
        dataset_mean: Calculated dataset mean (or None).
        dataset_std: Calculated dataset std (or None).

    Returns:
        True if upload was successful, False otherwise.
    """
    if not checkpoint_path or not os.path.exists(checkpoint_path):
        print(
            f"🚨 Error: Checkpoint path '{checkpoint_path}' not found or invalid. Skipping HF upload."
        )
        return False

    print(f"\n☁️ Attempting to upload model to Hugging Face Hub: {hf_repo_id}")
    temp_dir = "hf_upload_temp"  # Temporary directory for generated files
    os.makedirs(temp_dir, exist_ok=True)

    try:
        from huggingface_hub import HfApi
        from huggingface_hub import create_repo

        try:
            repo_url = create_repo(hf_repo_id, exist_ok=True, repo_type="model")
            print(f"   ✅ Repository '{hf_repo_id}' ensured: {repo_url}")

        except Exception as e:
            print(f"   🚨 Error creating/accessing repository: {e}")
            print(f"   Check your HF token and repo ID ('{hf_repo_id}').")
            return False

        config = get_hf_config(model_name, num_classes, id2label, label2id)
        config_path = os.path.join(temp_dir, "config.json")
        with open(config_path, "w") as f:
            json.dump(config, f, indent=2)
        print(f"   Generated config.json")

        input_size = metadata.get("input_size", model_configs[model_name]["input_size"])
        preprocessor_config = get_hf_preprocessor_config(
            model_name, input_size, dataset_mean, dataset_std
        )
        preprocessor_config_path = os.path.join(temp_dir, "preprocessor_config.json")
        with open(preprocessor_config_path, "w") as f:
            json.dump(preprocessor_config, f, indent=2)
        print(f"   Generated preprocessor_config.json")

        model_card_content = create_hf_model_card(
            model_name=model_name,
            dataset_name=dataset_name,
            num_classes=num_classes,
            id2label=id2label,
            metadata=metadata,
            dataset_mean=dataset_mean,
            dataset_std=dataset_std,
        )
        model_card_path = os.path.join(temp_dir, "README.md")
        with open(model_card_path, "w", encoding="utf-8") as f:
            f.write(model_card_content)
        print(f"   Generated README.md (Model Card)")

        api = HfApi()
        commit_message = f"Upload {model_name} trained on {dataset_name} (val_acc: {metadata.get('best_val_acc', 0.0):.4f})"

        api.upload_file(
            path_or_fileobj=checkpoint_path,
            path_in_repo="pytorch_model.bin",
            repo_id=hf_repo_id,
            commit_message=commit_message,
        )
        print(f"   Uploaded model weights (pytorch_model.bin)")

        api.upload_folder(
            folder_path=temp_dir,
            repo_id=hf_repo_id,
            commit_message="Upload config files and model card",
            ignore_patterns=["*.pt"],
        )
        print(f"   Uploaded config.json, preprocessor_config.json, README.md")

        print(f"✅ Model successfully uploaded to: {repo_url}")
        return True

    except ImportError:
        print("   🚨 Failed to upload: `huggingface_hub` library not found.")
        print("   Please install it: pip install huggingface_hub")
        return False

    except Exception as e:
        print(f"   🚨 An unexpected error occurred during Hugging Face upload: {e}")
        print(traceback.format_exc())
        return False

    finally:
        try:
            import shutil

            if os.path.exists(temp_dir):
                shutil.rmtree(temp_dir)

        except Exception as e:
            print(
                f"   ⚠️ Warning: Failed to clean up temporary directory '{temp_dir}': {e}"
            )


@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
@click.option(
    "--dataset-name",
    "-d",
    type=str,
    required=True,
    help="Hugging Face dataset name (e.g., 'keremberke/aerial-cactus').",
)
@click.option(
    "--hf-token",
    "-ht",
    type=str,
    default=lambda: os.environ.get(HF_TOKEN_ENV_VAR),
    help=f"Hugging Face API token (or set ${HF_TOKEN_ENV_VAR} env var). Required for upload.",
)
@click.option(
    "--hf-user",
    "-hu",
    type=str,
    default=lambda: os.environ.get(HF_USER_ENV_VAR),
    help=f"Hugging Face Username (or set ${HF_USER_ENV_VAR} env var). Required for upload.",
)
@click.option(
    "--wandb-token",
    "-wt",
    type=str,
    default=lambda: os.environ.get(WANDB_API_KEY_ENV_VAR),
    help=f"Weights & Biases API key (or set ${WANDB_API_KEY_ENV_VAR} env var). Required for logging.",
)
@click.option(
    "--wandb-project-name",
    "-wpn",
    type=str,
    required=True,
    help="WandB project name.",
)
@click.option(
    "--run-name",
    "-rn",
    type=str,
    required=True,
    help="Base name for the training run (used in WandB and filenames).",
)
@click.option(
    "--models",
    "-md",
    type=str,
    default="all",
    help=f'Comma-separated list of models to train (e.g., "resnet50,vit_b_16") or "all". Allowed names are: {", ".join(list(model_configs.keys()))}',
)
@click.option(
    "--epochs",
    "-e",
    type=int,
    default=30,
    show_default=True,
    help="Maximum number of training epochs.",
)
@click.option(
    "--training-mode",
    "-tm",
    type=click.Choice(list(training_mode_map.keys()), case_sensitive=False),
    default="sft",
    show_default=True,
    help="Training strategy: tl (head only), sft (selective unfreeze), fft (full fine-tune).",
)
@click.option(
    "--unfreeze-layers",
    "-ul",
    is_flag=True,
    help="If specified, layers mentioned in model_configs will be unfrozen in 'sft' mode. Will be discarded in other training modes.",
)
@click.option(
    "--dropout-p",
    "-drop",
    type=float,
    default=0.5,
    show_default=True,
    help="Dropout probability before the final classifier.",
)
@click.option(
    "--label-smoothing",
    "-ls",
    type=float,
    default=0.1,
    show_default=True,
    help="Label smoothing factor for CrossEntropyLoss (0.0 to disable).",
)
@click.option(
    "--calc-dataset-stats",
    is_flag=True,
    help="Calculate and use dataset-specific normalization stats instead of ImageNet defaults.",
)
@click.option(
    "--augment/--no-augment",
    "use_augmentation",
    default=True,
    help="Enable/disable training data augmentation (default: enabled).",
)
@click.option(
    "--use-mixed-precision/--no-mixed-precision",
    "use_mixed_precision",
    default=True,
    help="Enable/disable Automatic Mixed Precision (AMP) on CUDA (default: enabled).",
)
@click.option(
    "--metadata",
    "-m",
    type=str,
    default="",
    show_default=True,
    help="Optional custom metadata as key1=value1,key2=value2,...",
)
@click.option(
    "--seed",
    "-s",
    type=int,
    default=42,
    show_default=True,
    help="Random seed for reproducibility.",
)
@click.option(
    "--patience",
    "-p",
    type=int,
    default=10,
    show_default=True,
    help="Early stopping patience (epochs without validation accuracy improvement).",
)
@click.option(
    "--scheduler",
    "-sched",
    type=click.Choice(["cosine", "plateau", "none"], case_sensitive=False),
    default="cosine",
    show_default=True,
    help="Learning rate scheduler type ('none' to disable). Disabled automatically for 'tl' mode.",
)
@click.option(
    "--upload-to-hf/--no-upload-to-hf",
    "upload_to_hf",
    default=False,
    help="Upload the best model checkpoint from each run to HuggingFace Hub.",
)
@click.option(
    "--hf-repo-id",
    "-hfri",
    type=str,
    default="",
    help="HF repo ID (e.g., 'YourUser/YourModel'). Auto-generated if empty and upload is enabled.",
)
def train(
    dataset_name: str,
    hf_token: str | None,
    hf_user: str | None,
    wandb_token: str | None,
    wandb_project_name: str,
    run_name: str,
    models: str,
    epochs: int,
    training_mode: str,
    unfreeze_layers: str,
    dropout_p: float,
    label_smoothing: float,
    calc_dataset_stats: bool,
    use_augmentation: bool,
    use_mixed_precision: bool,
    metadata: str,
    seed: int,
    patience: int,
    scheduler: str,
    upload_to_hf: bool,
    hf_repo_id: str,
) -> None:
    """
    Main function to train aerial image classification models using PyTorch.
    Handles data loading, model initialization, training loop, logging, and optional HF upload.
    """
    print("=" * 60)
    print("🔥 Starting Aerial Image Classification Trainer")
    print("=" * 60)

    # Parameter Validation
    actual_training_mode = training_mode.lower()
    actual_scheduler = scheduler.lower()

    if actual_training_mode not in ["fft", "sft", "tl"]:
        raise ValueError(f' ⚠  An invalid training "{actual_training_mode}" mode has been specified.`')

    if actual_training_mode == "tl":
        print("Training mode is set to Transfer Learning. Only the classification head will be trainable.")

        if unfreeze_layers:
            unfreeze_layers = False
            print("⚠  Unfreeze_layers flag is specified but will be ignored.")

        if dropout_p > 0:
            print(f"⚠  A dropout_p value of {dropout_p} has been specified but be warned that the training mode is set to Transfer Learning.")

    elif actual_training_mode == "sft":
        print("Training mode is set to Selective Fine-tuning. Only the specified layers + the classification head will be trainable.")

        if not unfreeze_layers:
            unfreeze_layers = True
            print("⚠  Unfreeze_layers flag was not specified but will be set explicitly.")

    else:  # "fft"
        print("Training mode is set to Full Fine-tuning. All layers will be trainable.")

        if unfreeze_layers:
            unfreeze_layers = False
            print("⚠  Unfreeze_layers flag is specified but will be ignored.")

    # Mixed precision
    amp_available = is_cuda_available()
    actual_use_mixed_precision = use_mixed_precision and amp_available
    if use_mixed_precision and not amp_available:
        print("   ⚠️ Mixed precision requested but CUDA is not available. Disabling.")

    # Set reproducibility seed
    print(f"🌱 Setting random seed: {seed}")
    seed_worker = set_reproducibility(seed)
    g = torch.Generator()
    g.manual_seed(seed)

    # Hugging Face Login
    try:
        from huggingface_hub import login

        login(token=hf_token)
        print("Successfully logged in to Hugging Face Hub")
    except Exception as e:
        raise Exception(f"Failed to login to Hugging Face Hub.") from e

    # Setup Weights & Biases
    try:
        import wandb

        wandb.login(key=wandb_token)
        print("Successfully logged in to Weights & Biases")
    except Exception as e:
        raise Exception(f"Failed to login to Weights & Biases.") from e

    # Parse custom metadata
    metadata_dict = {}
    if metadata:
        try:
            for item in metadata.split(","):
                if "=" in item:
                    key, value = item.split("=", 1)
                    metadata_dict[key.strip()] = value.strip()
            print(f"   Parsed custom metadata: {metadata_dict}")
        except Exception as e:
            print(f"   ⚠️ Failed to parse metadata string '{metadata}': {e}")
            metadata_dict["raw_metadata"] = metadata

    # Dataset information
    print(f"📊 Loading info for dataset: {dataset_name}")
    try:
        # Use get_dataset_infos to handle potential configs
        ds_infos = get_dataset_infos(dataset_name)
        # Assume the first config if multiple exist
        ds_info = next(iter(ds_infos.values()))
        features = ds_info.features
        if "label" not in features or not hasattr(features["label"], "num_classes"):
            raise ValueError(
                "Dataset feature 'label' not found or is not a ClassLabel type."
            )
        num_classes = features["label"].num_classes
        class_names = features["label"].names
        id2label = {i: name for i, name in enumerate(class_names)}
        label2id = {name: i for i, name in enumerate(class_names)}
        print(f"   Number of classes: {num_classes}")
        print(f"   Class names: {class_names}")
    except Exception as e:
        raise Exception(
            f"🚨 Failed to load dataset info or infer classes for '{dataset_name}'. "
            f"Please ensure the dataset exists and has a 'label' feature of type ClassLabel."
        ) from e

    # Model selection
    models_to_train: list[str] = []
    if models.lower() == "all":
        models_to_train = list(model_configs.keys())

    else:
        requested_models = [m.strip() for m in models.split(",") if m.strip()]
        for model_key in requested_models:
            if model_key in model_configs:
                models_to_train.append(model_key)
            else:
                print(
                    f"   ⚠️ Model '{model_key}' not found in configurations. Skipping."
                )

    if not models_to_train:
        raise ValueError("🚨 No valid models selected to train. Exiting.")

    print(f"📦 Models selected for training: {', '.join(models_to_train)}")

    # Training loop for each model
    all_run_results: dict[str, dict[str, Any]] = {}  # Store results per model
    for model_name in models_to_train:
        print(f"\n{'=' * 20} Training Model: {model_name} {'=' * 20}")

        # Determine model-specific configs
        model_config = model_configs[model_name]
        batch_size = model_config["batch_size"]
        input_size = model_config["input_size"]
        model_lr = (
            model_config["base_lr_ft"]
            if actual_training_mode in ["fft", "sft"]
            else model_config["base_lr_tl"]
        )
        layers_to_unfreeze = model_config["unfreeze"] if unfreeze_layers else []
        print(f"   Learning Rate: {model_lr:.1E}")
        print(f"   Batch Size: {batch_size}")
        print(f"   Input Size: {input_size}x{input_size}")

        # Calculate dataset stats
        dataset_mean, dataset_std = None, None
        if calc_dataset_stats:
            dataset_mean, dataset_std = get_dataset_stats(
                dataset_name=dataset_name, 
                split="train",
                n_samples=2000,
                input_size=input_size,
            )
            if not dataset_mean or not dataset_std:
                print("   ⚠️ Failed to calculate dataset stats. Using ImageNet defaults.")

        try:
            model = initialize_model(
                model_name=model_name,
                num_classes=num_classes,
                requires_grad=(actual_training_mode == "fft"),
                layers_to_unfreeze=layers_to_unfreeze,
                dropout_p=dropout_p,
            )
        except Exception as e:
            raise Exception(f"🚨🚨 Failed to initialize model {model_name}.") from e

        # Load Data
        try:
            dataloaders_dict = load_data(
                dataset_name=dataset_name,
                input_size=input_size,
                batch_size=batch_size,
                seed_worker=seed_worker,
                g=g,
                use_augmentation=use_augmentation,
                dataset_mean=dataset_mean,
                dataset_std=dataset_std,
            )

        except Exception as e:
            if is_cuda_available():
                torch.cuda.empty_cache()
            raise Exception(f"🚨🚨 Failed to load data for {model_name}.") from e

        # Define loss, optimizer and scheduler
        # Loss function
        criterion = CrossEntropyLoss(label_smoothing=label_smoothing)
        ls_status = f"{label_smoothing}" if label_smoothing > 0 else "Disabled"
        print(f"   Loss Function: CrossEntropyLoss (Label Smoothing: {ls_status})")

        # Optimizer
        optimizer_name = ""
        optimizer: Optimizer
        if actual_training_mode == "tl":
            # Use Adam for head-only training (simpler, often effective)
            optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=model_lr)
            optimizer_name = "Adam"
            print(f"   Optimizer: {optimizer_name} (LR={model_lr:.1E})")

        else:
            # Use AdamW with weight decay for fine-tuning modes (fft, sft)
            # Apply weight decay only to weights (ndim >= 2), not biases or norms (ndim < 2)
            decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
            no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
            optimizer = AdamW(
                [
                    {
                        "params": decay_params,
                        "weight_decay": 0.05,
                    },  # Standard WD for weights
                    {
                        "params": no_decay_params,
                        "weight_decay": 0.0,
                    },  # No WD for biases/norms
                ],
                lr=model_lr,
                betas=(0.9, 0.999),
                eps=1e-8,
            )
            optimizer_name = "AdamW"
            print(f"   Optimizer: {optimizer_name} (LR={model_lr:.1E}, WD=0.05 on weights)")

        # Learning rate scheduler
        lr_scheduler: LRScheduler | None = None
        if actual_scheduler == "cosine":
            lr_scheduler = CosineAnnealingLR(optimizer, T_max=epochs, eta_min=model_lr / 100)  # Decay to 1/100th
            print(f"   Scheduler: CosineAnnealingLR (T_max={epochs}, eta_min={model_lr / 100:.1E})")

        elif actual_scheduler == "plateau":
            # Plateau scheduler steps based on validation metric (accuracy here)
            plateau_patience = max(1, patience // 2)  # Use half of early stopping patience
            lr_scheduler = ReduceLROnPlateau(
                optimizer,
                mode="max",
                factor=0.2,
                patience=plateau_patience,
                verbose=True,
            )
            print(f"   Scheduler: ReduceLROnPlateau (mode=max, factor=0.2, patience={plateau_patience})")

        else:  # 'none'
            print("   Scheduler: None")

        # Initialize WandB run
        wandb_run_name_full = (f"{model_name}-{run_name}-{actual_training_mode}")

        # Construct config dictionary for WandB
        run_config = {
            "model": model_name,
            "dataset": dataset_name,
            "epochs": epochs,
            "target_lr": model_lr,
            "batch_size": batch_size,
            "input_size": input_size,
            "seed": seed,
            "training_mode": actual_training_mode,
            "layers_to_unfreeze": layers_to_unfreeze,
            "dropout_p": dropout_p,
            "label_smoothing": label_smoothing,
            "data_augmentation": use_augmentation,
            "use_mixed_precision": actual_use_mixed_precision,
            "patience": patience,
            "scheduler": actual_scheduler,
            "data_normalization_strategy": "dataset" if calc_dataset_stats else "ImageNet",
            "dataset_mean": dataset_mean,
            "dataset_std": dataset_std,
            "optimizer": optimizer_name,
            **metadata_dict,  # custom metadata
        }

        try:
            if wandb.run is not None:
                print("   Finishing previous incomplete WandB run...")
                wandb.finish(exit_code=1, quiet=True)

            current_wandb_run = wandb.init(
                project=wandb_project_name,
                name=wandb_run_name_full,
                config=run_config,
                reinit=True,
                tags=[
                    model_name,
                    actual_training_mode,
                    dataset_name.split("/")[-1],
                    f"seed_{seed}",
                ],
            )
            print(f"   📊 WandB Run Initialized: {current_wandb_run.get_url()}")
            print("   Run Config:", json.dumps(run_config, indent=2, default=str))

        except Exception as e:
            raise Exception(f"   🚨 Failed to initialize W&B run for {model_name}.") from e

        # Run Training
        print(f"\n🚀 Starting training loop for {model_name}...")

        try:
            trained_model, model_results, best_checkpoint_path = train_model(
                model=model,
                dataloaders=dataloaders_dict,
                criterion=criterion,
                optimizer=optimizer,
                scheduler=lr_scheduler,
                num_epochs=epochs,
                patience=patience,
                model_name=f"{model_name}-{run_name}-{actual_training_mode}",
                use_mixed_precision=actual_use_mixed_precision,
            )
            all_run_results[model_name] = model_results

        except Exception as train_err:
            print(f"🚨🚨 Training loop failed for {model_name}: {train_err}")
            print(traceback.format_exc())
            all_run_results[model_name] = {"error": str(train_err)}  # Log error
            best_checkpoint_path = None  # No checkpoint if training failed
            if current_wandb_run:
                wandb.finish(exit_code=1)  # Mark run as failed

        # Upload to HuggingFace
        if upload_to_hf and best_checkpoint_path:
            model_repo_id = hf_repo_id
            if not model_repo_id:
                if not hf_user:
                    raise Exception(f"Huggingface username is invalid: {hf_user}.")

                model_repo_id = f"{hf_user}/aerial-{model_name.replace('_', '-')}-{run_name.replace('_', '-')}-{actual_training_mode}"
                print(f"   ⚠️ No --hf-repo-id provided. Using auto-generated ID: {model_repo_id}")

            # Prepare combined metadata for upload function
            upload_metadata = {**run_config, **model_results}

            upload_success = upload_model_to_hf(
                model_name=model_name,
                dataset_name=dataset_name,
                hf_repo_id=model_repo_id,
                checkpoint_path=best_checkpoint_path,
                num_classes=num_classes,
                id2label=id2label,
                label2id=label2id,
                metadata=upload_metadata,
                dataset_mean=dataset_mean,
                dataset_std=dataset_std,
            )

            if upload_success and current_wandb_run:
                # Log HF model URL to the completed WandB run
                hf_model_url = f"https://huggingface.co/{model_repo_id}"
                wandb.log({"huggingface_model_url": hf_model_url})
                print(f"   Logged HF URL to WandB: {hf_model_url}")

        elif upload_to_hf and not best_checkpoint_path:
            print("   ⚠️ Skipping HuggingFace upload because no best checkpoint was saved (training might not have improved or failed).")

        # Finish WandB run & cleanup
        if current_wandb_run:
            print("   Finishing WandB run for this model...")
            wandb.finish(exit_code=0 if best_checkpoint_path else 1, quiet=True)  # Mark success/failure

        # Explicitly delete large objects to free memory before next loop iteration
        del model, optimizer, criterion, dataloaders_dict
        if "lr_scheduler" in locals() and lr_scheduler:
            del lr_scheduler
        if is_cuda_available():
            torch.cuda.empty_cache()

    # Print results table
    table_str = "\n" + ("=" * 60)
    table_str += "\n🏁 OVERALL TRAINING SUMMARY 🏁\n"
    table_str += ("=" * 60)
    table_str += f"\nDataset: {dataset_name}"
    table_str += f"\nSeed Used: {seed}"
    table_str += f"\nTraining Mode(s) Attempted: {training_mode} (effective modes may vary per run)\n"

    table_data = []
    models_completed = 0

    for model_key in models_to_train:
        results_dict = all_run_results.get(model_key)
        
        if results_dict and "error" not in results_dict:
            acc = results_dict.get("best_val_acc", 0.0) * 100
            epoch = results_dict.get("best_epoch", -1)
            time_mins = results_dict.get("train_time", 0.0) / 60
            status = "✅ Success" if epoch != -1 else "⚠️ No Improve"
            table_data.append([model_key, f"{acc:.2f}%", epoch, f"{time_mins:.1f}", status])
            models_completed += 1

        elif results_dict and "error" in results_dict:
            status = f"🚨 Error: {results_dict['error'][:30]}..."
            table_data.append([model_key, "N/A", "N/A", "N/A", status])

        else:
            table_data.append([model_key, "N/A", "N/A", "N/A", "❓ Unknown"])

    # Define headers
    headers = ["Model", "Best Val Acc", "Best Epoch", "Train Time (m)", "Status"]
    table_str += ("\n" + tabulate(table_data, headers=headers, tablefmt="github"))
    print("\n" + table_str)

    # Save to file
    output_file = f"training_results_{run_name}_seed{seed}_{actual_training_mode}.md"
    with open(output_file, "w", encoding="utf-8") as f:
        f.write(table_str)

    # Save Summary Results to File
    summary_filename = f"training_summary_{run_name}_seed{seed}.json"
    summary_data = {
        "script_args": {k: v for k, v in click.get_current_context().params if "token" in k},  # Log CLI args
        "dataset": dataset_name,
        "seed": seed,
        "models_attempted": models_to_train,
        "models_completed": models_completed,
        "results_per_model": all_run_results,
    }
    try:
        with open(summary_filename, "w") as f:
            json.dump(summary_data, f, indent=2, default=str)  # Use default=str for non-serializable types

        print(f"\n📄 Overall summary saved to '{summary_filename}'")
    except Exception as e:
        print(f"   🚨 Failed to save summary JSON to file: {e}")

    print("\n✨ Trainer script finished. ✨\n")


if __name__ == "__main__":
    train()
