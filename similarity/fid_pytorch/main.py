import torch
from torcheval.metrics import FrechetInceptionDistance
from PIL import Image
import os
from torchvision import transforms
from torch.utils.data import DataLoader, Dataset

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

class ImageCountMismatchError(Exception):
    def __init__(self, message="Number of real and synthetic images are not equal"):
        super().__init__(message)

class ImageDataset(Dataset):
    def __init__(self, image_dir, resize_size=(299, 299)):
        self.image_dir = image_dir
        self.resize_size = resize_size
        self.image_paths = [os.path.join(image_dir, img_name) for img_name in os.listdir(image_dir)]

    def __len__(self):
        return len(self.image_paths)

    def __getitem__(self, idx):
        img_path = self.image_paths[idx]
        try:
            image = Image.open(img_path).convert("RGB")
            image = image.resize(self.resize_size)  # Resize for InceptionV3
            image = transforms.ToTensor()(image)  # Convert to tensor
            return image
        except Exception as e:
            print(f"Skipping {img_path} due to error: {e}")
            return None

def compute_fid(ref_dir: str, eval_dir: str, batch_size: int = 32) -> float:
    print(f"Device: {device}")

    real_dataset = ImageDataset(ref_dir)
    synthetic_dataset = ImageDataset(eval_dir)

    if len(real_dataset) != len(synthetic_dataset):
        raise ImageCountMismatchError(f"Warning: Real images ({len(real_dataset)}) and synthetic images ({len(synthetic_dataset)}) have different counts.")

    real_loader = DataLoader(real_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)
    synthetic_loader = DataLoader(synthetic_dataset, batch_size=batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Initialize FID metric and move it to GPU (if available)
    fid = FrechetInceptionDistance().to(device)

    # Process real images in batches
    for real_batch in real_loader:
        if real_batch is None:
            continue
        real_batch = real_batch.to(device)
        fid.update(real_batch, is_real=True)

    # Process synthetic images in batches
    for synthetic_batch in synthetic_loader:
        if synthetic_batch is None:
            continue
        synthetic_batch = synthetic_batch.to(device)
        fid.update(synthetic_batch, is_real=False)

    # Compute FID
    fid_score = fid.compute()
    
    return fid_score