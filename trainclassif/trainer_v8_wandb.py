import json
import os
import random
import time
import traceback
from collections import Counter
from collections import defaultdict
from datetime import datetime
from typing import Any
from typing import Literal
from typing import TypedDict

import numpy as np
import torch
import wandb
from datasets import get_dataset_infos
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix
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
from torchvision.transforms import v2 as transforms
from tqdm.auto import tqdm

load_dotenv("../.env")

HF_TOKEN_ENV_VAR = "HF_TOKEN"
HF_USER_ENV_VAR = "HF_USER"
WANDB_API_KEY_ENV_VAR = "WANDB_API_KEY"
HF_TOKEN = os.environ.get(HF_TOKEN_ENV_VAR)
HF_USER = os.environ.get(HF_USER_ENV_VAR)
WANDB_TOKEN = os.environ.get(WANDB_API_KEY_ENV_VAR)

if not HF_TOKEN or not HF_USER or not WANDB_TOKEN:
    raise ValueError("Required .env vars were not found or loaded properly")

TRAINING_MODE_MAP = {
    "tl": "Transfer Learning (Head Only)",
    "sft": "Selective Fine-Tuning",
    "fft": "Full Fine-Tuning",
}
DEFAULT_SEED = 42
SUPPORTED_MODELS = [
    "mobilenet_v2",
    "resnet50",
    "efficientnet_b2",
    "efficientnetv2_m",
    "vit_b_16",
]

# types
AugmentationLevel = Literal["none", "basic", "medium", "advanced"]
TrainingMode = Literal["sft", "fft", "tl"]
ModelArchitecture = Literal[
    "mobilenet_v2", "resnet50", "efficientnet_b2", "efficientnetv2_m", "vit_b_16"
]
SchedulerType = Literal["none", "cosine", "plateau"]


class TrainingConfig(TypedDict):
    dataset_name: str
    model: ModelArchitecture
    epochs: int
    input_size: int
    batch_size: int
    training_mode: TrainingMode
    learning_rate: float
    augment_level: AugmentationLevel
    weight_decay: float
    layers_to_unfreeze: list[str]
    dropout_p: float
    label_smoothing: float
    calc_dataset_stats: bool
    use_mixed_precision: bool
    seed: int
    patience: int
    scheduler: SchedulerType
    cutmix_or_mixup: bool
    cutmix_alpha: float
    mixup_alpha: float


def set_reproducibility(seed: int = DEFAULT_SEED) -> callable:
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

    return seed_worker


def get_balanced_sample(dataset, num_classes, n_samples_per_class):
    class_samples = defaultdict(list)
    total_needed = n_samples_per_class * num_classes

    for example in dataset:
        label = example["label"]
        if len(class_samples[label]) < n_samples_per_class:
            class_samples[label].append(example)
        if sum(len(v) for v in class_samples.values()) >= total_needed:
            break

    balanced_samples = [item for sublist in class_samples.values() for item in sublist]
    return balanced_samples


def get_dataset_stats(
    seed: int,
    dataset_name: str,
    split: str,
    n_samples_per_class: int,
    input_size: int,
    num_classes: int,
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
        f"⏳ Calculating approximate dataset stats for '{dataset_name}' [{split}] "
        f"split using {n_samples_per_class * num_classes} samples..."
    )
    mean = torch.zeros(3)
    std = torch.zeros(3)
    count = 0

    try:
        # Use streaming to avoid downloading the full dataset upfront
        dataset = load_dataset(dataset_name, split=split, streaming=True)

        # Shuffle and take a sample for calculation
        shuffled_dataset = dataset.shuffle(seed=seed)
        sampled_dataset = get_balanced_sample(
            dataset=shuffled_dataset,
            num_classes=num_classes,
            n_samples_per_class=n_samples_per_class,
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
            sampled_dataset,
            total=len(sampled_dataset),
            desc="Calculating Stats",
            leave=False,
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
                print(f"⚠️ Skipping image during stats calc due to error: {e}")
                continue

        if count == 0:
            print("🚨 No valid images processed for stats calculation.")
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
    augment_level: AugmentationLevel,
    dataset_mean: list | None,
    dataset_std: list | None,
) -> dict[str, DataLoader]:
    """Loads and preprocesses the dataset with optional augmentation
    and custom normalization and returns a dictionary containing
    'train' and 'val' DataLoaders."""

    print(f"🔃 Loading dataset '{dataset_name}'...")
    dataset = load_dataset(dataset_name)

    print("Train label distribution:", Counter(dataset["train"]["label"]))
    print("Val label distribution:", Counter(dataset["validation"]["label"]))
    print("Unique train labels:", sorted(set(dataset["train"]["label"])))
    print("Unique val labels:", sorted(set(dataset["validation"]["label"])))


    if dataset_mean and dataset_std:
        print("Using dataset-specific normalization")
        normalize = transforms.Normalize(mean=dataset_mean, std=dataset_std)
    else:
        print("Using ImageNet default normalization.")
        normalize = transforms.Normalize(
            mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]
        )

    # Resize slightly larger than input size for cropping
    resize_size = int(input_size * 1.143)  # e.g., 256 for 224 input

    val_transform = transforms.Compose(
        [
            transforms.Resize(resize_size),
            transforms.CenterCrop(input_size),
            transforms.ToTensor(),
            normalize,
        ]
    )

    print(f"Preparing {augment_level} train augmentations...")
    if augment_level == "advanced":
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
    elif augment_level == "medium":
        train_transform = transforms.Compose(
            [
                transforms.RandomResizedCrop(
                    input_size, scale=(0.8, 1.0)
                ),  # Less aggressive crop
                transforms.RandomAffine(
                    degrees=10, translate=(0.05, 0.05), shear=5
                ),  # Milder affine
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),  # Optional
                transforms.RandomRotation(10),  # Milder rotation
                transforms.ColorJitter(
                    brightness=0.1, contrast=0.1, saturation=0.1, hue=0.05
                ),  # Milder color jitter
                transforms.ToTensor(),
                normalize,
            ]
        )
    elif augment_level == "basic":
        train_transform = transforms.Compose(
            [
                transforms.Resize(resize_size),
                transforms.CenterCrop(
                    input_size
                ),  # Or RandomResizedCrop(input_size, scale=(0.9, 1.0))
                transforms.RandomHorizontalFlip(),
                transforms.RandomVerticalFlip(),  # Optional
                transforms.ToTensor(),
                normalize,
            ]
        )
    else:
        train_transform = val_transform

    def _ensure_rgb(image):
        """Converts image to RGB if not already."""
        if not hasattr(image, "convert"):
            try:
                image = transforms.ToPILImage()(image)
            except Exception as e:
                raise TypeError(f"Error converting to PIL. Type: {type(image)}. {e}")
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
            raise Exception(f"🚨 Error processing image in {split} set.") from e

    print("Applying transformations...")
    dataset["train"] = dataset["train"].map(
        lambda ex: transform_dataset(ex, "train"),
        batched=False,
        desc="Applying train transforms",
        load_from_cache_file=False,
    )
    dataset["validation"] = dataset["validation"].map(
        lambda ex: transform_dataset(ex, "val"),
        batched=False,
        desc="Applying validation transforms",
        load_from_cache_file=False,
    )

    dataset.set_format(type="torch", columns=["image", "label"])

    # create DataLoaders
    num_workers = os.cpu_count() // 2 if os.cpu_count() else 4
    print(f"Using {num_workers} workers for DataLoaders.")
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
            drop_last=False,  # was True originally
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
                raise AttributeError(f"Cannot determine classifier head structure")

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
                param.requires_grad = True
    else:
        raise ValueError("⚠️ Could not verify classifier trainability.")


def initialize_model(
    model_name: str,
    num_classes: int,
    requires_grad: bool,
    dropout_p: float,
    layers_to_unfreeze: list[str],
) -> Module:
    """Initializes the model, sets gradient requirements based
    on training mode and returns the initialized model"""

    print(f"Initializing model: {model_name}")
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

    # selective unfreezing for SFT
    if layers_to_unfreeze:
        print(f"Unfreezing layers with keywords: {layers_to_unfreeze}")
        unfrozen_count = 0

        for name, param in model.named_parameters():
            if any(layer_key in name for layer_key in layers_to_unfreeze):
                if not param.requires_grad:
                    param.requires_grad = True
                unfrozen_count += 1

        print(f"Unfroze {unfrozen_count} parameters matching keywords.")

    num_trainable = sum(p.numel() for p in model.parameters() if p.requires_grad)
    num_total = sum(p.numel() for p in model.parameters())
    perc = num_trainable / num_total
    print(f"Model: {model_name}")
    print(f"Trainable Parameters: {num_trainable:,} / {num_total:,} ({perc:.2%})")

    return model


def train_model(
    model: Module,
    dataloaders: dict[str, DataLoader],
    criterion: Module,
    optimizer: Optimizer,
    scheduler: LRScheduler | None,
    num_epochs: int,
    patience: int,
    use_mixed_precision: bool,
    class_names: list[str],
    cutmix_or_mixup: transforms.RandomChoice | None,
) -> tuple[Module, dict[str, Any]]:
    """
    Trains the model for a specified number of epochs with early stopping.

    Logs epoch-level metrics and final best metrics to WandB.
    Calculates and logs the confusion matrix from the best validation epoch.
    Does NOT save checkpoints locally or to WandB artifacts.
    Does NOT load best weights back into the model after training.

    Returns a tuple containing:
        - The model with the best validation accuracy weights loaded.
        - A dictionary containing training history and results.
    """

    start_time = time.time()
    val_acc_history = []
    best_epoch = -1
    best_acc = 0.0
    best_f1 = 0.0
    best_loss = float("inf")
    best_val_preds: list[int] = []
    best_val_labels: list[int] = []

    results: dict[str, Any] = {
        "best_val_acc": 0.0,
        "best_val_loss": float("inf"),
        "best_val_f1": 0.0,
        "best_epoch": -1,
        "val_acc_history": [],
        "val_loss_history": [],
        "train_acc_history": [],
        "train_loss_history": [],
        "lr_history": [],
        "train_time": 0.0,
        "confusion_matrix": None,
    }

    dev = device("cuda:0" if is_cuda_available() else "cpu")
    model.to(dev)
    print(f"🏋️‍♀️ Starting training on device: {dev}")

    # initialize gradient scaler for mixed
    # precision if enabled and available
    scaler = None
    amp_enabled = use_mixed_precision and dev.type == "cuda"
    if amp_enabled:
        try:
            from torch.amp import GradScaler
            from torch.amp import autocast

            scaler = GradScaler()
            print("⚡ Automatic Mixed Precision (AMP) Enabled.")
        except ImportError:
            print("⚠️ Warning: torch.amp not available. Disabling Mixed Precision.")
            amp_enabled = False  # disable if import fails
    else:
        print(
            f"⚠️ Automatic mixed precision: {'Disabled' if not use_mixed_precision else 'Unavailable (not on CUDA)'}."
        )

    # epoch progress bar
    epoch_pbar = tqdm(range(num_epochs), desc="Epochs", position=0, leave=True)
    epochs_no_improve = 0  # counter for early stopping

    for epoch in epoch_pbar:
        epoch_metrics: dict[str, float] = {}  # Store metrics for this epoch
        current_lr = optimizer.param_groups[0]["lr"]  # Get current LR
        results["lr_history"].append(current_lr)

        for phase in ["train", "val"]:
            is_train = phase == "train"
            model.train() if is_train else model.eval()

            running_loss = 0.0
            running_corrects = 0
            all_preds = []
            all_labels = []

            # batch progress bar
            batch_pbar = tqdm(
                dataloaders[phase],
                desc=f"{phase.capitalize():<5}",  # e.g., "Train", "Val  "
                position=1,  # inner bar
                leave=False,  # remove after completion
                total=len(dataloaders[phase]),
            )

            for batch in batch_pbar:
                inputs = batch["image"].to(dev, non_blocking=True)
                labels = batch["label"].to(dev, non_blocking=True)

                # cutmix/mixup
                if is_train and cutmix_or_mixup:
                    inputs, labels = cutmix_or_mixup(inputs, labels)

                # zero gradients before forward pass
                optimizer.zero_grad(set_to_none=True)  # more memory efficient

                # forward pass with autocasting for AMP if enabled
                with set_grad_enabled(is_train):  # enable grads only in training
                    if amp_enabled:
                        from torch.amp import autocast  # import locally for clarity

                        with autocast(
                            device_type="cuda:0" if is_cuda_available() else "cpu"
                        ):
                            outputs = model(inputs)
                            loss = criterion(outputs, labels)
                    else:  # default precision
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # get predictions (index of max logit)
                    _, preds = torch.max(outputs, 1)

                # backward pass + optimize only if in training phase
                if is_train:
                    if scaler:  # using AMP scaler
                        scaler.scale(loss).backward()  # scale loss
                        # optional: gradient clipping (unscale first)
                        # scaler.unscale_(optimizer)
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        scaler.step(optimizer)  # optimizer step (unscales internally)
                        scaler.update()  # update scaler for next iteration
                    else:  # default precision
                        loss.backward()
                        # optional: gradient clipping
                        # torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
                        optimizer.step()

                # statistics update
                batch_loss = loss.item()
                running_loss += batch_loss * inputs.size(
                    0
                )  # accumulate loss scaled by batch size
                running_corrects += torch.sum(
                    preds == labels.data
                ).item()  # accumulate correct predictions

                # store predictions and labels for epoch-level metrics (F1, etc.)
                all_preds.extend(preds.cpu().numpy())
                all_labels.extend(labels.cpu().numpy())

                # update batch progress bar postfix to show batch loss and LR
                batch_pbar.set_postfix(
                    loss=f"{batch_loss:.4f}",
                    lr=f"{current_lr:.1E}",
                )

            # calculate epoch metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects / len(dataloaders[phase].dataset)
            epoch_metrics[f"{phase}_loss"] = epoch_loss
            epoch_metrics[f"{phase}_acc"] = epoch_acc
            results[f"{phase}_loss_history"].append(epoch_loss)
            results[f"{phase}_acc_history"].append(epoch_acc)

            # calculate Precision, Recall, F1 (macro average)
            # zero_division=0 handles cases where a class might not be predicted in a batch/epoch
            epoch_precision = precision_score(
                all_labels, all_preds, average="macro", zero_division=0
            )
            epoch_recall = recall_score(
                all_labels, all_preds, average="macro", zero_division=0
            )
            epoch_f1 = f1_score(all_labels, all_preds, average="macro", zero_division=0)
            epoch_metrics[f"{phase}_precision"] = epoch_precision
            epoch_metrics[f"{phase}_recall"] = epoch_recall
            epoch_metrics[f"{phase}_f1"] = epoch_f1

            # validation phase specific actions
            if phase == "val":
                val_acc_history.append(epoch_acc)  # keep track for early stopping check

                # step the scheduler based on validation metric (if applicable)
                if scheduler is not None:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(epoch_acc)  # step based on validation accuracy

                    else:
                        scheduler.step()  # CosineAnnealingLR steps every epoch regardless of metrics

                # check for improvement & early stopping
                if epoch_acc > best_acc:
                    best_acc = epoch_acc
                    best_loss = epoch_loss
                    best_epoch = epoch
                    best_f1 = epoch_f1
                    epochs_no_improve = 0  # reset counter
                    best_val_preds = all_preds.copy()
                    best_val_labels = all_labels.copy()
                    tqdm.write(f"✅ Epoch {epoch}: Val Acc improved to {best_acc:.4f}.")
                else:
                    epochs_no_improve += 1
                    tqdm.write(
                        f"📉 Epoch {epoch}: Val Acc ({epoch_acc:.4f}) did not improve from best ({best_acc:.4f}). ({epochs_no_improve}/{patience})"
                    )

        # log epoch metrics to WandB safely
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
                tqdm.write(f"🚨 WandB logging failed for epoch {epoch}: {e}")

        # update epoch progress bar postfix with key metrics
        epoch_pbar.set_postfix(
            TrL=f"{epoch_metrics.get('train_loss', 0):.3f}",
            TrA=f"{epoch_metrics.get('train_acc', 0):.3f}",
            VL=f"{epoch_metrics.get('val_loss', 0):.3f}",
            VA=f"{epoch_metrics.get('val_acc', 0):.3f}",
            BestVA=f"{best_acc:.3f}",
            LR=f"{current_lr:.1E}",
        )

        # print epoch summary to console
        tqdm.write(
            f"Epoch {epoch}/{num_epochs-1} -> "
            f"Train[L:{epoch_metrics.get('train_loss', 0):.4f} A:{epoch_metrics.get('train_acc', 0):.4f} F1:{epoch_metrics.get('train_f1', 0):.4f}] | "
            f"Val[L:{epoch_metrics.get('val_loss', 0):.4f} A:{epoch_metrics.get('val_acc', 0):.4f} F1:{epoch_metrics.get('val_f1', 0):.4f}] | "
            f"LR: {current_lr:.6f}"
        )

        # early stopping check
        if epochs_no_improve >= patience:
            tqdm.write(
                f"\n⏳ Early stopping triggered at epoch {epoch} after {patience} epochs with no improvement."
            )
            break  # exit epoch loop

    end_time = time.time()
    total_time = end_time - start_time
    results["train_time"] = total_time
    print(f"\n🏁 Training finished in {total_time // 60:.0f}m {total_time % 60:.1f}s")

    # log results if an improvement was found
    if best_epoch != -1:
        print(f"🏆 Best Validation Acc: {best_acc:.4f} achieved at Epoch {best_epoch}.")
        results["best_val_acc"] = best_acc
        results["best_val_loss"] = best_loss
        results["best_val_f1"] = best_f1
        results["best_epoch"] = best_epoch

        try:
            conf_matrix = confusion_matrix(best_val_labels, best_val_preds)
            results["confusion_matrix"] = conf_matrix.tolist()
            print("Confusion Matrix calculated from best validation epoch.")

            # log confusion matrix plot and best metrics to WandB
            if wandb.run:
                try:
                    # log confusion matrix plot
                    wandb.log(
                        {
                            "confusion_matrix_best_val_epoch": wandb.plot.confusion_matrix(
                                preds=best_val_preds,
                                y_true=best_val_labels,
                                class_names=class_names,
                            )
                        }
                    )
                    print("Confusion Matrix plot logged to WandB.")
                except Exception as e:
                    print(f"🚨 Failed to log confusion matrix to WandB: {e}")

        except Exception as e:
            print(f"🚨 Error calculating or storing confusion matrix: {e}")
            results["confusion_matrix"] = None

    else:
        print("⚠️ No improvement in validation accuracy was observed during training.")
        results["best_val_acc"] = val_acc_history[0] if val_acc_history else 0.0
        results["best_val_loss"] = (
            results["val_loss_history"][0]
            if results["val_loss_history"]
            else float("inf")
        )
        results["best_val_f1"] = 0.0
        results["best_epoch"] = best_epoch

    if wandb.run:
        try:
            wandb.summary["best_val_acc"] = best_acc
            wandb.summary["best_val_loss"] = best_loss
            wandb.summary["best_val_f1"] = best_f1
            wandb.summary["best_epoch"] = best_epoch
            print("✅ Final best metrics logged to WandB summary.")
        except Exception as e:
            print(f"🚨 Failed to log final best metrics to WandB summary: {e}")

    results["val_acc_history"] = val_acc_history

    return model, results


def train() -> None:
    """
    Main function to train aerial image classification models using PyTorch.
    This version is designed to work with WandB sweeping feature.
    """
    
    print("\n")
    print("-" * 60)
    print("🔥 Starting Aerial Image Classification Trainer")
    print("-" * 60)

    # HF login
    try:
        from huggingface_hub import login

        login(token=HF_TOKEN)
    except Exception as e:
        raise Exception(f"Failed to login to HF.") from e

    # init wandb run
    try:
        wandb_run = wandb.init(job_type="train")
        print(f"👟 WandB Run Initialized: {wandb_run.get_url()}")
        print("Run Config:", json.dumps(wandb.config, indent=2, default=str))
    except Exception as e:
        raise Exception(f"🚨 Failed to initialize W&B run.") from e

    # wandb config
    config = TrainingConfig(
        dataset_name=wandb.config["dataset_name"],
        model=wandb.config["model"],
        epochs=wandb.config["epochs"],
        input_size=wandb.config["input_size"],
        batch_size=wandb.config["batch_size"],
        training_mode=wandb.config["training_mode"],
        learning_rate=wandb.config["learning_rate"],
        augment_level=wandb.config["augment_level"],
        weight_decay=wandb.config["weight_decay"],
        layers_to_unfreeze=wandb.config["layers_to_unfreeze"],
        dropout_p=wandb.config["dropout_p"],
        label_smoothing=wandb.config["label_smoothing"],
        calc_dataset_stats=wandb.config["calc_dataset_stats"],
        use_mixed_precision=wandb.config["use_mixed_precision"],
        seed=wandb.config["seed"],
        patience=wandb.config["patience"],
        scheduler=wandb.config["scheduler"],
        cutmix_or_mixup=wandb.config.get("cutmix_or_mixup", False),
        cutmix_alpha=wandb.config.get("cutmix_alpha", 1.0),
        mixup_alpha=wandb.config.get("mixup_alpha", 0.2)
    )

    # model
    if not config["model"] or config["model"] not in SUPPORTED_MODELS:
        raise ValueError("🚨 Unsupported model specified.")

    # layers_to_unfreeze
    if config["training_mode"] not in ["fft", "sft", "tl"]:
        raise ValueError(f"⚠️ Invalid training mode: {config['training_mode']}")

    if config["training_mode"] == "tl":
        print(f"Training mode = TL. Only the head will be trainable.")

        if config["layers_to_unfreeze"]:
            config["layers_to_unfreeze"] = []
            print("⚠️ layers_to_unfreeze will be ignored.")

        if config["dropout_p"] > 0:
            print(f"⚠️ dropout_p ({config['dropout_p']}) specified with TL.")

    elif config["training_mode"] == "sft":
        print("Training mode = SFT. Only specified layers + head will be trainable.")

        if not config["layers_to_unfreeze"]:
            raise ValueError("⚠️ layers_to_unfreeze not specified.")

    else:  # "fft"
        print("Training mode = FFT. All layers will be trainable.")

        if config["layers_to_unfreeze"]:
            config["layers_to_unfreeze"] = []
            print("⚠️ layers_to_unfreeze will be ignored.")

    # dropout_p
    if config["dropout_p"] < 0 or config["dropout_p"] > 1:
        raise ValueError("🚨 Dropout value must be a valid percentage.")

    # mixed precision
    amp_available = is_cuda_available()
    actual_use_mixed_precision = config["use_mixed_precision"] and amp_available
    if config["use_mixed_precision"] and not amp_available:
        print("⚠️ Mixed precision requested but CUDA is not available.")

    # set seed
    print(f"🌱 Setting random seed: {config['seed']}")
    seed_worker = set_reproducibility(config["seed"])
    g = torch.Generator()
    g.manual_seed(config["seed"])

    # dataset info
    try:
        ds_infos = get_dataset_infos(config["dataset_name"])
        ds_info = next(iter(ds_infos.values()))
        features = ds_info.features
        if "label" not in features or not hasattr(features["label"], "num_classes"):
            raise ValueError("Dataset feature 'label' not found.")

        num_classes = features["label"].num_classes
        class_names = features["label"].names
        id2label = {i: name for i, name in enumerate(class_names)}
        label2id = {name: i for i, name in enumerate(class_names)}
        print(f"Number of classes: {num_classes}\nClass names: {class_names}")

    except Exception as e:
        raise Exception(
            f"🚨 Failed to load dataset info or infer classes for '{config['dataset_name']}'. "
            f"Please ensure the dataset exists and has a 'label' feature of type ClassLabel."
        ) from e

    # dataset stats
    dataset_mean, dataset_std = None, None
    if config["calc_dataset_stats"]:
        dataset_mean, dataset_std = get_dataset_stats(
            seed=config["seed"],
            dataset_name=config["dataset_name"],
            split="train",
            n_samples_per_class=100,
            input_size=config["input_size"],
            num_classes=num_classes,
        )

    try:
        dataloaders_dict = load_data(
            dataset_name=config["dataset_name"],
            input_size=config["input_size"],
            batch_size=config["batch_size"],
            seed_worker=seed_worker,
            g=g,
            augment_level=config["augment_level"],
            dataset_mean=dataset_mean,
            dataset_std=dataset_std,
        )
    except Exception as e:
        if is_cuda_available():
            torch.cuda.empty_cache()
        raise Exception(f"🚨 Failed to load data.") from e

    try:
        model = initialize_model(
            model_name=config["model"],
            num_classes=num_classes,
            requires_grad=(config["training_mode"] == "fft"),
            layers_to_unfreeze=config["layers_to_unfreeze"],
            dropout_p=config["dropout_p"],
        )
    except Exception as e:
        raise Exception(f"🚨 Failed to initialize {config['model']}.") from e

    # loss function
    criterion = CrossEntropyLoss(label_smoothing=config["label_smoothing"])
    print("Loss Func: CrossEntropyLoss")
    print(f"Label Smoothing: {config['label_smoothing']}")

    # optimizer
    optimizer: Optimizer
    optimizer_name = ""
    if (
        config["training_mode"] == "tl"
    ):  # Use Adam for head-only training (simpler, often effective)
        optimizer = Adam(
            filter(lambda p: p.requires_grad, model.parameters()),
            lr=config["learning_rate"],
        )
        optimizer_name = "Adam"
        print(f"Optimizer: {optimizer_name} (LR={config['learning_rate']:.1E})")
    else:
        # Use AdamW with weight decay for fine-tuning modes (fft, sft)
        # Apply weight decay only to weights (ndim >= 2), not biases or norms (ndim < 2)
        decay_params = [
            p for p in model.parameters() if p.requires_grad and p.ndim >= 2
        ]
        no_decay_params = [
            p for p in model.parameters() if p.requires_grad and p.ndim < 2
        ]
        optimizer = AdamW(
            [
                {
                    "params": decay_params,
                    "weight_decay": config["weight_decay"],
                },  # Standard WD for weights
                {
                    "params": no_decay_params,
                    "weight_decay": 0.0,
                },  # No WD for biases/norms
            ],
            lr=config["learning_rate"],
            betas=(0.9, 0.999),
            eps=1e-8,
        )
        optimizer_name = "AdamW"
        print(f"Optimizer: {optimizer_name} (LR={config['learning_rate']:.1E}")
        print(f"Weight decay = {config['weight_decay']} on weights)")

    # lr scheduler
    lr_scheduler: LRScheduler | None = None
    if config["scheduler"] == "cosine":
        lr_scheduler = CosineAnnealingLR(
            optimizer, T_max=config["epochs"], eta_min=config["learning_rate"] / 100
        )  # Decay to 1/100th
        print("Scheduler: CosineAnnealingLR")
        print(f"T_max={config['epochs']}")
        print(f"eta_min={config['learning_rate'] / 100:.1E})")

    elif config["scheduler"] == "plateau":
        # plateau scheduler steps based on accuracy
        # used half of early stopping patience
        plateau_patience = max(1, config["patience"] // 2)
        lr_scheduler = ReduceLROnPlateau(
            optimizer,
            mode="max",
            factor=0.2,
            patience=plateau_patience,
            verbose=True,
        )
        print("Scheduler: ReduceLROnPlateau")
        print(f"mode=max, factor=0.2, patience={plateau_patience})")

    else:  # 'none'
        print("Scheduler: None")

    wandb.config.update(
        {
            "actual_use_mixed_precision": actual_use_mixed_precision,
            "cuda": is_cuda_available(),
            "num_classes": num_classes,
            "data_normalization_strategy": (
                "dataset" if dataset_mean and dataset_std else "ImageNet"
            ),
            "dataset_mean": dataset_mean,
            "dataset_std": dataset_std,
            "optimizer": optimizer_name,
            "cutmix_alpha": config["cutmix_alpha"],
            "mixup_alpha": config["mixup_alpha"]
        },
        allow_val_change=True,
    )

    # cutmix/mixup
    if config["cutmix_or_mixup"]:
        cutmix = transforms.CutMix(num_classes=num_classes, alpha=config["cutmix_alpha"])
        mixup = transforms.MixUp(num_classes=num_classes, alpha=config["mixup_alpha"])
        cutmix_or_mixup = transforms.RandomChoice([cutmix, mixup])
    else:
        cutmix_or_mixup = None

    print(f"🚀 Starting training loop for {config['model']}...")
    try:
        best_model, model_results = train_model(
            model=model,
            dataloaders=dataloaders_dict,
            criterion=criterion,
            optimizer=optimizer,
            scheduler=lr_scheduler,
            num_epochs=config["epochs"],
            patience=config["patience"],
            use_mixed_precision=actual_use_mixed_precision,
            class_names=class_names,
            cutmix_or_mixup=cutmix_or_mixup,
        )

        # print results table
        table_str = "\n🏁 Training Summary\n"
        table_str += "-" * 20
        table_str += f"\n"
        table_str += (
            f"```json\n{json.dumps(wandb.run.config, indent=2, default=str)}\n\n"
        )

        table_data = []

        if model_results:
            acc = model_results.get("best_val_acc", 0.0) * 100
            epoch = model_results.get("best_epoch", -1)
            time_mins = model_results.get("train_time", 0.0) / 60
            status = "✅ Success" if epoch != -1 else "⚠️ No Improve"
            table_data.append(
                [config["model"], f"{acc:.2f}%", epoch, f"{time_mins:.1f}", status]
            )

        else:
            table_data.append([config["model"], "N/A", "N/A", "N/A", "❓ Unknown"])

        # table headers
        headers = ["Model", "Best Val Acc", "Best Epoch", "Train Time (m)", "Status"]
        table_str += "\n" + tabulate(table_data, headers=headers, tablefmt="github")
        print("\n" + table_str)

        # save to file
        date_str = datetime.now().strftime("%Y_%m_%d")
        output_file = f"{date_str}_res_{wandb.run.name}_{config['training_mode']}.md"
        with open(output_file, "w", encoding="utf-8") as f:
            f.write(table_str)

        print("\n✨ Trainer finished. ✨\n")

        # finish WandB run & cleanup
        if wandb_run:
            print(" Finishing WandB run for this model...")
            wandb.finish(exit_code=0, quiet=True)

        if is_cuda_available():
            torch.cuda.empty_cache()

    except Exception as e:
        print(f"🚨 Training loop failed for {config['model']}")
        print(traceback.format_exc())

        if wandb_run:
            wandb.finish(exit_code=1)
        raise


if __name__ == "__main__":
    train()
