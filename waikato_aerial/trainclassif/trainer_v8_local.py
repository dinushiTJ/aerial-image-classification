from copy import deepcopy
import json
import os
import random
import time
import traceback
from collections import Counter, defaultdict
from typing import Any, Literal

import click
import numpy as np
import torch
import wandb
from datasets import get_dataset_infos
from datasets import load_dataset
from dotenv import load_dotenv
from sklearn.metrics import confusion_matrix, f1_score, precision_score, recall_score
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
from torch.utils.data import default_collate
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
DEFAULT_BATCH_SIZE = 32
CALC_DATASET_STATS = False
DEFAULT_INPUT_SIZE = 224
DEFAULT_PATIENCE = 10
CUTMIX_OR_MIXUP = True
CUTMIX_ALPHA = 1.0
MIXUP_ALPHA = 0.2
USE_MIXED_PRECISION = True
AUGMENT_LEVEL = "none"
DEFAULT_EPOCHS = 50
DATASETS = {
    "dushj98/aerial_real_only",
    "dushj98/aerial_real_plus_0010",
    "dushj98/aerial_real_plus_0025",
    "dushj98/aerial_real_plus_0050",
    "dushj98/aerial_real_plus_0075",
    "dushj98/aerial_real_plus_0100",
    "dushj98/aerial_real_plus_0125",
    "dushj98/aerial_real_plus_0150",
}
DATASET_SUFFIX_MAP = {
    "dushj98/aerial_real_only": "real",
    "dushj98/aerial_real_plus_0010": "p10",
    "dushj98/aerial_real_plus_0025": "p25",
    "dushj98/aerial_real_plus_0050": "p50",
    "dushj98/aerial_real_plus_0075": "p75",
    "dushj98/aerial_real_plus_0100": "p100",
    "dushj98/aerial_real_plus_0125": "p125",
    "dushj98/aerial_real_plus_0150": "p150",
}
PRE_DEFINED_MODEL_CONFIGS: dict[str, dict[str, dict]] = {
    "vit_b_16": {
        "tl": {
            "dropout_p": 0.2,
            "learning_rate": 5e-04,
            "label_smoothing": 0.0,
            "weight_decay": 0.13,
            "scheduler": "cosine",
            "layers_to_unfreeze": ["encoder.layers.encoder_layer_8.", "encoder.layers.encoder_layer_9.", "encoder.layers.encoder_layer_10.", "encoder.layers.encoder_layer_11."],
        },
        "sft": {
            "dropout_p": 0.4,
            "learning_rate": 1e-04,
            "label_smoothing": 0.2,
            "weight_decay": 0.06,
            "scheduler": "cosine",
            "layers_to_unfreeze": ["encoder.layers.encoder_layer_8.", "encoder.layers.encoder_layer_9.", "encoder.layers.encoder_layer_10.", "encoder.layers.encoder_layer_11."],
        },
        "fft": {
            "dropout_p": 0.5,
            "learning_rate": 5e-06,
            "label_smoothing": 0.0,
            "weight_decay": 0.06,
            "scheduler": "plateau",
            "layers_to_unfreeze": ["encoder.layers.encoder_layer_8.", "encoder.layers.encoder_layer_9.", "encoder.layers.encoder_layer_10.", "encoder.layers.encoder_layer_11."],
        },
    },
    "resnet50": {
        "tl": {
            "dropout_p": 0.5,
            "learning_rate": 3e-04,
            "label_smoothing": 0.2,
            "weight_decay": 0.02,
            "scheduler": "plateau",
            "layers_to_unfreeze": ["layer4."],
        },
        "sft": {
            "dropout_p": 0.2,
            "learning_rate": 7e-06,
            "label_smoothing": 0.0,
            "weight_decay": 0.19,
            "scheduler": "plateau",
            "layers_to_unfreeze": ["layer4."],
        },
        "fft": {
            "dropout_p": 0.4,
            "learning_rate": 7e-06,
            "label_smoothing": 0.0,
            "weight_decay": 0.04,
            "scheduler": "plateau",
            "layers_to_unfreeze": ["layer4."],
        },
    },
    "efficientnet_b2": {
        "tl": {
            "dropout_p": 0.1,
            "learning_rate": 2e-04,
            "label_smoothing": 0.0,
            "weight_decay": 0.02,
            "scheduler": "plateau",
            "layers_to_unfreeze": ["features.6.", "features.7.", "features.8."],
        },
        "sft": {
            "dropout_p": 0.4,
            "learning_rate": 4e-05,
            "label_smoothing": 0.2,
            "weight_decay": 0.14,
            "scheduler": "cosine",
            "layers_to_unfreeze": ["features.6.", "features.7.", "features.8."],
        },
        "fft": {
            "dropout_p": 0.3,
            "learning_rate": 1e-05,
            "label_smoothing": 0.2,
            "weight_decay": 0.03,
            "scheduler": "plateau",
            "layers_to_unfreeze": ["features.6.", "features.7.", "features.8."],
        },
    },
}

# types
AugmentationLevel = Literal["none", "basic", "medium", "advanced"]
TrainingMode = Literal["sft", "fft", "tl"]
ModelArchitecture = Literal[
    "mobilenet_v2", "resnet50", "efficientnet_b2", "efficientnetv2_m", "vit_b_16"
]
SchedulerType = Literal["none", "cosine", "plateau"]


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
    cutmix_or_mixup: transforms.Transform | None,
    num_classes: int,
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

    if cutmix_or_mixup:
        def collate_fn(batch):
            collated_batch = default_collate(batch)
            images = collated_batch["image"]
            labels = collated_batch["label"] # labels are currently torch.long

            # Convert integer labels to one-hot encoded float tensor
            # num_classes is available from the outer scope
            labels_one_hot = torch.nn.functional.one_hot(labels, num_classes=num_classes).float()

            # Apply CutMix or MixUp
            return cutmix_or_mixup(images, labels_one_hot) # Pass the one-hot encoded labels

        collate_func = collate_fn

    else:
        collate_func = None

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
            collate_fn=collate_func,
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
    cutmix_or_mixup: bool,
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

    # initialize gradient scaler for mixed precision if enabled and available
    scaler = None
    amp_enabled = use_mixed_precision and dev.type == "cuda"
    if amp_enabled:
        try:
            from torch.amp import GradScaler, autocast
            scaler = GradScaler()
            print("⚡ Automatic Mixed Precision (AMP) Enabled.")
        except ImportError:
            print("⚠️ Warning: torch.amp not available. Disabling Mixed Precision.")
            amp_enabled = False  # disable if import fails
    else:
        print(f"⚠️ Automatic mixed precision: {'Disabled' if not use_mixed_precision else 'Unavailable (not on CUDA)'}.")

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
            
            # Check if we should calculate standard classification metrics in this phase/configuration
            calculate_standard_metrics = not is_train or not cutmix_or_mixup

            # batch progress bar
            batch_pbar = tqdm(
                dataloaders[phase],
                desc=f"{phase.capitalize():<5}",  # e.g., "Train", "Val  "
                position=1,  # inner bar
                leave=False,  # remove after completion
                total=len(dataloaders[phase]),
            )

            for batch in batch_pbar:
                # Dataloader for train might return (images, one_hot_labels) tuple
                # Dataloader for val will return dictionary {"image": images, "label": integer_labels}
                if (isinstance(batch, tuple) or isinstance(batch, list)) and len(batch) == 2:
                    # This happens during training if collate_fn is used
                     inputs, labels = batch # labels will be one_hot float tensors
                else:
                    # This happens during validation, or training without custom collate_fn
                    inputs = batch["image"]
                    labels = batch["label"] # labels will be integer long tensors

                inputs = inputs.to(dev, non_blocking=True)
                labels = labels.to(dev, non_blocking=True) # Labels on device, type depends on phase/collate_fn

                # zero gradients before forward pass
                optimizer.zero_grad(set_to_none=True)  # more memory efficient

                # forward pass with autocasting for AMP if enabled
                with set_grad_enabled(is_train):  # enable grads only in training
                    if amp_enabled:
                        from torch.amp import autocast  # import locally for clarity

                        with autocast(device_type="cuda:0" if is_cuda_available() else "cpu"):
                            outputs = model(inputs)
                            # Loss calculation works with either integer or one_hot/mixed labels
                            loss = criterion(outputs, labels)
                    else:  # default precision
                        outputs = model(inputs)
                        loss = criterion(outputs, labels)

                    # get predictions (index of max logit)
                    # This always gives integer class indices, regardless of label format
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
                running_loss += batch_loss * inputs.size(0)  # accumulate loss scaled by batch size

                # Standard Metric Calculation: Only if we should calculate them (validation or train without mixup)
                if calculate_standard_metrics:
                    # Use the original integer labels for metric calculation if available
                    # During validation, 'labels' is already the integer tensor
                    # During training without mixup, 'labels' is already the integer tensor
                    running_corrects += torch.sum(preds == labels).item()

                    # Store predicted integer class indices and ground truth integer labels
                    all_preds.extend(preds.cpu().numpy())
                    all_labels.extend(labels.cpu().numpy()) # These are the original integer labels from the batch

                # update batch progress bar postfix to show batch loss and LR
                batch_pbar.set_postfix(loss=f"{batch_loss:.4f}",lr=f"{current_lr:.1E}",)

            # calculate epoch metrics
            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_metrics[f"{phase}_loss"] = epoch_loss
            results[f"{phase}_loss_history"].append(epoch_loss)

            # Calculate standard classification metrics only if we collected data for them
            if calculate_standard_metrics:
                 epoch_acc = running_corrects / len(dataloaders[phase].dataset)
                 epoch_metrics[f"{phase}_acc"] = epoch_acc
                 results[f"{phase}_acc_history"].append(epoch_acc)

                 # calculate Precision, Recall, F1 (macro average)
                 try:
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
                 except Exception as e:
                     tqdm.write(f"⚠️ Error calculating metrics for {phase}: {e}")
                     epoch_metrics[f"{phase}_precision"] = 0.0
                     epoch_metrics[f"{phase}_recall"] = 0.0
                     epoch_metrics[f"{phase}_f1"] = 0.0

            else:
                 # If standard metrics are not calculated (train with mixup), log placeholder values
                 epoch_metrics[f"{phase}_acc"] = 0.0
                 epoch_metrics[f"{phase}_precision"] = 0.0
                 epoch_metrics[f"{phase}_recall"] = 0.0
                 epoch_metrics[f"{phase}_f1"] = 0.0
                 # Append placeholder to history to maintain list length consistency if needed, or handle separately
                 results[f"{phase}_acc_history"].append(0.0)

            # validation phase specific actions
            if phase == "val":
                val_acc_history.append(epoch_metrics["val_acc"]) # Use the calculated validation accuracy

                # step the scheduler based on validation metric (if applicable)
                if scheduler is not None:
                    if isinstance(scheduler, ReduceLROnPlateau):
                        scheduler.step(epoch_metrics["val_acc"]) # step based on validation accuracy

                    else: # CosineAnnealingLR
                        scheduler.step() # CosineAnnealingLR steps every epoch

                # check for improvement & early stopping using validation accuracy
                # Note: We always use validation accuracy for early stopping criteria
                current_val_acc = epoch_metrics["val_acc"]
                if current_val_acc > best_acc:
                    best_acc = current_val_acc
                    best_loss = epoch_metrics["val_loss"]
                    best_epoch = epoch
                    best_f1 = epoch_metrics["val_f1"]
                    epochs_no_improve = 0  # reset counter
                    # Store predictions and labels collected ONLY from the validation phase
                    best_val_preds = all_preds.copy() # These are from validation, so they are correct integer labels
                    best_val_labels = all_labels.copy() # These are from validation, so they are correct integer labels
                    tqdm.write(f"✅ Epoch {epoch}: Val Acc improved to {best_acc:.4f}.")
                else:
                    epochs_no_improve += 1
                    tqdm.write(
                        f"📉 Epoch {epoch}: Val Acc ({current_val_acc:.4f}) did not improve from best ({best_acc:.4f}). ({epochs_no_improve}/{patience})"
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
            TrL=f"{epoch_metrics.get('train_loss', 0.0):.3f}",
            TrA=f"{epoch_metrics.get('train_acc', 0.0):.3f}",
            VL=f"{epoch_metrics.get('val_loss', 0.0):.3f}",
            VA=f"{epoch_metrics.get('val_acc', 0.0):.3f}",
            BestVA=f"{best_acc:.3f}",
            LR=f"{current_lr:.1E}",
        )

        # print epoch summary to console
        tqdm.write(
            f"Epoch {epoch}/{num_epochs-1} -> "
            f"Train[L:{epoch_metrics.get('train_loss', 0.0):.4f} A:{epoch_metrics.get('train_acc', 0.0):.4f} F1:{epoch_metrics.get('train_f1', 0.0):.4f}] | "
            f"Val[L:{epoch_metrics.get('val_loss', 0.0):.4f} A:{epoch_metrics.get('val_acc', 0.0):.4f} F1:{epoch_metrics.get('val_f1', 0.0):.4f}] | "
            f"LR: {current_lr:.6f}"
        )

        # early stopping check
        if epochs_no_improve >= patience:
            tqdm.write(f"\n⏳ Early stopping triggered at epoch {epoch} after {patience} epochs with no improvement.")
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
        # Log the metrics from the last epoch if no improvement
        last_epoch_metrics = {}
        if len(results["val_acc_history"]) > 0:
             results["best_val_acc"] = results["val_acc_history"][-1]
             results["best_val_loss"] = results["val_loss_history"][-1]
             # F1, Precision, Recall from the last epoch if available, otherwise 0.0
             # This assumes the last epoch's metrics are stored in epoch_metrics before the loop ends
             # A safer way is to store epoch_metrics in a list per epoch
             # For simplicity here, we might just log 0 or the last recorded non-zero if desired
             results["best_val_f1"] = epoch_metrics.get("val_f1", 0.0) if 'val_f1' in epoch_metrics else 0.0


        results["best_epoch"] = best_epoch


    if wandb.run:
        try:
            # Log best validation metrics to summary
            wandb.summary["best_val_acc"] = results["best_val_acc"]
            wandb.summary["best_val_loss"] = results["best_val_loss"]
            wandb.summary["best_val_f1"] = results["best_val_f1"]
            wandb.summary["best_epoch"] = results["best_epoch"]
            print("✅ Final best metrics logged to WandB summary.")
        except Exception as e:
            print(f"🚨 Failed to log final best metrics to WandB summary: {e}")

    # You might want to return the model state dict of the best epoch,
    # but your original code didn't, so we'll stick to that.
    # If you wanted to return the best model, you'd need to save and load state_dict.

    results["val_acc_history"] = val_acc_history

    return model, results # Returns the model in its final state after all epochs (not necessarily the best)



@click.command(context_settings=dict(help_option_names=["-h", "--help"]))
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
    
    # WandB login
    try:
        import wandb

        wandb.login(key=WANDB_TOKEN)
        print("Successfully logged in to Weights & Biases")
    except Exception as e:
        raise Exception(f"Failed to login to Weights & Biases.") from e


    # loop over datasets
    for dataset_name in DATASETS:        
        # set seed
        print(f"🌱 Setting random seed: {DEFAULT_SEED}")
        seed_worker = set_reproducibility(DEFAULT_SEED)
        g = torch.Generator()
        g.manual_seed(DEFAULT_SEED)

        # dataset info
        try:
            ds_infos = get_dataset_infos(dataset_name)
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
                f"🚨 Failed to load dataset info or infer classes for '{dataset_name}'. "
                f"Please ensure the dataset exists and has a 'label' feature of type ClassLabel."
            ) from e

        # dataset stats
        dataset_mean, dataset_std = None, None
        if CALC_DATASET_STATS:
            dataset_mean, dataset_std = get_dataset_stats(
                seed=DEFAULT_SEED,
                dataset_name=dataset_name,
                split="train",
                n_samples_per_class=100,
                input_size=DEFAULT_INPUT_SIZE,
                num_classes=num_classes,
            )

        # cutmix/mixup
        if CUTMIX_OR_MIXUP:
            cutmix = transforms.CutMix(num_classes=num_classes, alpha=CUTMIX_ALPHA)
            mixup = transforms.MixUp(num_classes=num_classes, alpha=MIXUP_ALPHA)
            cutmix_or_mixup = transforms.RandomChoice([cutmix, mixup])
        else:
            cutmix_or_mixup = None
        
        try:
            dataloaders_dict = load_data(
                dataset_name=dataset_name,
                input_size=DEFAULT_INPUT_SIZE,
                batch_size=DEFAULT_BATCH_SIZE,
                seed_worker=seed_worker,
                g=g,
                augment_level=AUGMENT_LEVEL,
                dataset_mean=dataset_mean,
                dataset_std=dataset_std,
                cutmix_or_mixup=cutmix_or_mixup,
                num_classes=num_classes,
            )
        except Exception as e:
            if is_cuda_available():
                torch.cuda.empty_cache()
            raise Exception(f"🚨 Failed to load data.") from e
        
        # loop over models
        for model_name in PRE_DEFINED_MODEL_CONFIGS.keys():
            # skipping already trained models
            if model_name == "vit_b_16":
                print("Skipping 'vit_b_16' model as it is already trained.")
                continue

            if model_name == "resnet50" and dataset_name in ["dushj98/aerial_real_plus_0010", "dushj98/aerial_real_plus_0025"]:
                print(f"Skipping {dataset_name} for {model_name} as it is already trained.")
                continue

            # loop over training modes
            for training_mode in TRAINING_MODE_MAP.keys():
                training_configs = deepcopy(PRE_DEFINED_MODEL_CONFIGS[model_name][training_mode])

                # validate configs
                if training_mode == "tl":
                    print(f"Training mode = TL. Only the head will be trainable.")

                    if training_configs["layers_to_unfreeze"]:
                        training_configs["layers_to_unfreeze"] = []
                        print("⚠️ layers_to_unfreeze will be ignored.")

                    if training_configs["dropout_p"] > 0:
                        print(f"⚠️ dropout_p ({training_configs['dropout_p']}) specified with TL.")

                elif training_mode == "sft":
                    print("Training mode = SFT. Only specified layers + head will be trainable.")

                    if not training_configs["layers_to_unfreeze"]:
                        raise ValueError("⚠️ layers_to_unfreeze not specified.")

                else:  # "fft"
                    print("Training mode = FFT. All layers will be trainable.")

                    if training_configs["layers_to_unfreeze"]:
                        training_configs["layers_to_unfreeze"] = []
                        print("⚠️ layers_to_unfreeze will be ignored.")

                # dropout_p
                if training_configs["dropout_p"] < 0 or training_configs["dropout_p"] > 1:
                    raise ValueError("🚨 Dropout value must be a valid percentage.")

                # mixed precision
                amp_available = is_cuda_available()
                actual_use_mixed_precision = USE_MIXED_PRECISION and amp_available
                if USE_MIXED_PRECISION and not amp_available:
                    print("⚠️ Mixed precision requested but CUDA is not available.")
                
                # initialize model
                try:
                    model = initialize_model(
                        model_name=model_name,
                        num_classes=num_classes,
                        requires_grad=(training_mode == "fft"),
                        layers_to_unfreeze=training_configs["layers_to_unfreeze"],
                        dropout_p=training_configs["dropout_p"],
                    )
                except Exception as e:
                    raise Exception(f"🚨 Failed to initialize {model_name}.") from e
                
                # loss function
                criterion = CrossEntropyLoss(label_smoothing=training_configs["label_smoothing"])
                print("Loss Func: CrossEntropyLoss")
                print(f"Label Smoothing: {training_configs['label_smoothing']}")

                # optimizer
                optimizer: Optimizer
                optimizer_name = ""
                if training_mode == "tl":  # Use Adam for head-only training (simpler, often effective)
                    optimizer = Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=training_configs["learning_rate"])
                    optimizer_name = "Adam"
                    print(f"Optimizer: {optimizer_name} (LR={training_configs['learning_rate']:.1E})")
                else:
                    # Use AdamW with weight decay for fine-tuning modes (fft, sft)
                    # Apply weight decay only to weights (ndim >= 2), not biases or norms (ndim < 2)
                    decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim >= 2]
                    no_decay_params = [p for p in model.parameters() if p.requires_grad and p.ndim < 2]
                    optimizer = AdamW(
                        [
                            {
                                "params": decay_params,
                                "weight_decay": training_configs["weight_decay"],
                            },  # Standard WD for weights
                            {
                                "params": no_decay_params,
                                "weight_decay": 0.0,
                            },  # No WD for biases/norms
                        ],
                        lr=training_configs["learning_rate"],
                        betas=(0.9, 0.999),
                        eps=1e-8,
                    )
                    optimizer_name = "AdamW"
                    print(f"Optimizer: {optimizer_name} (LR={training_configs['learning_rate']:.1E}")
                    print(f"Weight decay = {training_configs['weight_decay']} on weights)")
                
                # lr scheduler
                lr_scheduler: LRScheduler | None = None
                if training_configs["scheduler"] == "cosine":
                    lr_scheduler = CosineAnnealingLR(
                        optimizer, T_max=DEFAULT_EPOCHS, eta_min=training_configs["learning_rate"] / 100
                    )  # Decay to 1/100th
                    print("Scheduler: CosineAnnealingLR")
                    print(f"T_max={DEFAULT_EPOCHS}")
                    print(f"eta_min={training_configs['learning_rate'] / 100:.1E})")

                elif training_configs["scheduler"] == "plateau":
                    # plateau scheduler steps based on accuracy
                    # used half of early stopping patience
                    plateau_patience = max(1, DEFAULT_PATIENCE // 2)
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

                # update training configs
                training_configs["model"] = model_name
                training_configs["training_mode"] = training_mode
                training_configs["dataset_name"] = dataset_name
                training_configs["num_classes"] = num_classes
                training_configs["augment_level"] = AUGMENT_LEVEL
                training_configs["cutmix_or_mixup"] = CUTMIX_OR_MIXUP
                training_configs["cutmix_alpha"] = CUTMIX_ALPHA
                training_configs["mixup_alpha"] = MIXUP_ALPHA
                training_configs["calc_dataset_stats"] = CALC_DATASET_STATS
                training_configs["dataset_mean"] = dataset_mean
                training_configs["dataset_std"] = dataset_std
                training_configs["seed"] = DEFAULT_SEED
                training_configs["use_mixed_precision"] = USE_MIXED_PRECISION
                training_configs["patience"] = DEFAULT_PATIENCE
                training_configs["input_size"] = DEFAULT_INPUT_SIZE
                training_configs["batch_size"] = DEFAULT_BATCH_SIZE
                training_configs["epochs"] = DEFAULT_EPOCHS
                training_configs["cuda"] = is_cuda_available()
                training_configs["actual_use_mixed_precision"] = actual_use_mixed_precision
                training_configs["optimizer"] = optimizer_name
                training_configs["data_normalization_strategy"] = "dataset" if dataset_mean and dataset_std else "ImageNet"

                # init wandb run
                dataset_suffix = DATASET_SUFFIX_MAP[dataset_name]
                wandb_project_name = f"{model_name}-{num_classes}cls-{dataset_suffix}-{training_mode}"

                try:
                    if wandb.run is not None:
                        print(f"Finishing previous incomplete WandB run {wandb.run.name}...")
                        wandb.finish(exit_code=1, quiet=True)

                    wandb_run = wandb.init(
                        project=wandb_project_name,
                        config=training_configs,
                        reinit=True,
                        tags=[
                            model_name,
                            training_mode,
                            dataset_name.split("/")[-1],
                            f"seed_{DEFAULT_SEED}",
                        ],
                    )
                    print(f"👟 WandB Run Initialized: {wandb_run.get_url()}")
                    print("Run Config:", json.dumps(training_mode, indent=2, default=str))

                except Exception as e:
                    raise Exception(f"🚨 Failed to initialize W&B run for {model_name}.") from e

                print(f"🚀 Starting training loop for {model_name}...")
                try:
                    best_model, model_results = train_model(
                        model=model,
                        dataloaders=dataloaders_dict,
                        criterion=criterion,
                        optimizer=optimizer,
                        scheduler=lr_scheduler,
                        num_epochs=DEFAULT_EPOCHS,
                        patience=DEFAULT_PATIENCE,
                        use_mixed_precision=actual_use_mixed_precision,
                        class_names=class_names,
                        cutmix_or_mixup=True if cutmix_or_mixup else False,
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
                        table_data.append([model_name, f"{acc:.2f}%", epoch, f"{time_mins:.1f}", status])

                    else:
                        table_data.append([model_name, "N/A", "N/A", "N/A", "❓ Unknown"])

                    # table headers
                    headers = ["Model", "Best Val Acc", "Best Epoch", "Train Time (m)", "Status"]
                    table_str += "\n" + tabulate(table_data, headers=headers, tablefmt="github")
                    print("\n" + table_str)

                    # # save to file
                    # date_str = datetime.now().strftime("%Y_%m_%d")
                    # output_file = f"{date_str}_res_{wandb.run.name}_{config['training_mode']}.md"
                    # with open(output_file, "w", encoding="utf-8") as f:
                    #     f.write(table_str)

                    print(f"\n✨ Trainer finished for {model_name} ({dataset_suffix}). ✨\n")

                    # finish WandB run & cleanup
                    if wandb_run:
                        print(" Finishing WandB run for this model...")
                        wandb.finish(exit_code=0, quiet=True)

                    if is_cuda_available():
                        torch.cuda.empty_cache()

                except Exception as e:
                    print(f"🚨 Training loop failed for {model_name} ({dataset_suffix})")
                    print(traceback.format_exc())

                    if wandb_run:
                        wandb.finish(exit_code=1)
                    raise
    
    print("\n✨ Trainer finished. ✨\n")

if __name__ == "__main__":
    train()
