# src/dataset.py
"""
Dataset utilities for face recognition metric learning.
Supports map-style loading with class-balanced sampling.
"""
from pathlib import Path
from typing import Optional, Callable, List, Tuple, Iterator
import torch
from torch.utils.data import Dataset, Sampler
from PIL import Image
from torchvision import transforms
import numpy as np
from collections import defaultdict
from loguru import logger
import random


class FaceDataset(Dataset):
    """
    Map-style dataset for face recognition.
    
    Expected directory structure:
        root/{split}/{label}/{image}.jpg
    
    Args:
        root: Root directory containing split subdirectories
        split: One of 'train', 'val', 'test'
        transform: Optional torchvision transforms
        min_images_per_class: Minimum images required per identity
        labels_to_use: Optional whitelist of labels to include
        embedding_mode: If True, returns only images (no labels) for inference
    """
    
    SUPPORTED_EXTENSIONS = {'.jpg', '.jpeg', '.png', '.bmp', '.webp'}
    
    def __init__(
        self,
        root: str,
        split: str = "train",
        transform: Optional[Callable] = None,
        min_images_per_class: int = 3,
        labels_to_use: Optional[List[str]] = None,
        embedding_mode: bool = False,
    ):
        self.root = Path(root) / split
        self.split = split
        self.embedding_mode = embedding_mode
        
        # Default transforms for 112x112 face images
        if transform is None:
            if split == "train":
                self.transform = transforms.Compose([
                    transforms.RandomHorizontalFlip(p=0.5),
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                       std=[0.5, 0.5, 0.5]),
                ])
            else:
                self.transform = transforms.Compose([
                    transforms.ToTensor(),
                    transforms.Normalize(mean=[0.5, 0.5, 0.5], 
                                       std=[0.5, 0.5, 0.5]),
                ])
        else:
            self.transform = transform
        
        self.samples: List[Tuple[Path, int]] = []
        self.class_names: List[str] = []
        self.label_to_idx: dict = {}
        
        self._scan_dataset(min_images_per_class, labels_to_use)
        
        # Build class-to-indices mapping for balanced sampling
        self.class_to_indices: dict[int, List[int]] = defaultdict(list)
        if not embedding_mode:
            for idx, (_, class_idx) in enumerate(self.samples):
                self.class_to_indices[class_idx].append(idx)
        
        logger.info(
            f"📦 {split}: {len(self.samples)} images, "
            f"{len(self.class_names)} classes"
        )
    
    def _scan_dataset(self, min_images: int, whitelist: Optional[List[str]]):
        """Scan directory structure and populate self.samples"""
        if not self.root.exists():
            raise ValueError(f"Directory not found: {self.root}")
        
        for class_dir in sorted(self.root.iterdir()):
            if not class_dir.is_dir():
                continue
            
            label = class_dir.name
            if whitelist and label not in whitelist:
                continue
            
            # Collect supported image files
            images = [
                p for p in class_dir.iterdir() 
                if p.suffix.lower() in self.SUPPORTED_EXTENSIONS
            ]
            
            if len(images) < min_images:
                continue
            
            # Register class
            if label not in self.label_to_idx:
                class_idx = len(self.class_names)
                self.label_to_idx[label] = class_idx
                self.class_names.append(label)
            else:
                class_idx = self.label_to_idx[label]
            
            # Add samples
            for img_path in sorted(images):
                self.samples.append((img_path, class_idx))
    
    def __len__(self) -> int:
        return len(self.samples)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, ...]:
        img_path, label = self.samples[idx]
        
        try:
            image = Image.open(img_path).convert("RGB")
        except Exception as e:
            logger.warning(f"Failed to load {img_path}: {e}")
            # Return random valid sample as fallback
            return self.__getitem__(random.randint(0, len(self) - 1))
        
        if self.transform:
            image = self.transform(image)
        
        if self.embedding_mode:
            return image, str(img_path)  # Return path for reference
        
        return image, label
    
    def get_class_weights(self) -> torch.Tensor:
        """Compute inverse frequency weights for class-balanced loss"""
        if self.embedding_mode:
            return torch.ones(len(self.class_names))
        
        counts = torch.zeros(len(self.class_names))
        for _, class_idx in self.samples:
            counts[class_idx] += 1
        # Inverse frequency with smoothing
        weights = 1.0 / (counts + 1.0)
        return weights / weights.sum() * len(weights)


class ClassBalancedBatchSampler(Sampler[List[int]]):
    """
    Sampler for metric learning: P classes × K instances per batch.
    
    Ensures each batch contains multiple images from the same identities,
    which is crucial for triplet/arcface losses.
    
    Args:
        dataset: FaceDataset instance
        batch_size: Total batch size (must be divisible by instances_per_class)
        instances_per_class: Number of samples per identity in batch (K)
        shuffle: Whether to shuffle classes between epochs
        seed: Random seed for reproducibility
    """
    
    def __init__(
        self,
        dataset: FaceDataset,
        batch_size: int,
        instances_per_class: int = 4,
        shuffle: bool = True,
        seed: int = 42,
    ):
        if dataset.embedding_mode:
            raise ValueError("ClassBalancedBatchSampler requires labeled dataset")
        
        self.dataset = dataset
        self.batch_size = batch_size
        self.K = instances_per_class
        self.P = batch_size // instances_per_class  # classes per batch
        self.shuffle = shuffle
        self.rng = np.random.RandomState(seed)
        
        if batch_size % instances_per_class != 0:
            raise ValueError(
                f"batch_size ({batch_size}) must be divisible by "
                f"instances_per_class ({instances_per_class})"
            )
        
        if self.P > len(dataset.class_names):
            logger.warning(
                f"Requested P={self.P} classes but only "
                f"{len(dataset.class_names)} available. Adjusting P."
            )
            self.P = len(dataset.class_names)
            self.batch_size = self.P * self.K
    
    def __iter__(self) -> Iterator[List[int]]:
        class_indices = list(self.dataset.class_to_indices.keys())
        
        while True:
            if self.shuffle:
                self.rng.shuffle(class_indices)
            
            # Yield batches of P classes × K instances
            for start in range(0, len(class_indices), self.P):
                selected_classes = class_indices[start:start + self.P]
                if len(selected_classes) < self.P:
                    break
                
                batch_indices = []
                for cls in selected_classes:
                    available = self.dataset.class_to_indices[cls]
                    if len(available) >= self.K:
                        # Sample without replacement
                        choices = self.rng.choice(
                            available, size=self.K, replace=False
                        )
                    else:
                        # Fallback: sample with replacement
                        choices = self.rng.choice(
                            available, size=self.K, replace=True
                        )
                    batch_indices.extend(choices.tolist())
                
                if len(batch_indices) == self.batch_size:
                    yield batch_indices
    
    def __len__(self) -> int:
        """Approximate number of batches per epoch"""
        return len(self.dataset) // self.batch_size


def create_dataloaders(
    cfg: dict,
    data_root: str,
) -> Tuple[DataLoader, Optional[DataLoader]]:
    """
    Factory function to create train/val dataloaders from config.
    
    Returns:
        train_loader, val_loader (val may be None if not configured)
    """
    from torch.utils.data import DataLoader
    
    # Train dataset with balanced sampler
    train_dataset = FaceDataset(
        root=data_root,
        split="train",
        min_images_per_class=cfg["dataset"]["min_images_per_id"],
    )
    
    train_sampler = ClassBalancedBatchSampler(
        train_dataset,
        batch_size=cfg["training"]["batch_size"],
        instances_per_class=cfg["training"].get("instances_per_class", 4),
        shuffle=True,
        seed=cfg["dataset"].get("seed", 42),
    )
    
    train_loader = DataLoader(
        train_dataset,
        batch_sampler=train_sampler,
        num_workers=cfg["training"].get("num_workers", 4),
        pin_memory=torch.backends.mps.is_available(),
        persistent_workers=cfg["training"].get("num_workers", 4) > 0,
    )
    
    # Validation dataset (simple random sampler)
    val_loader = None
    if cfg["training"].get("val_split", True):
        val_dataset = FaceDataset(
            root=data_root,
            split="val",
            min_images_per_class=cfg["dataset"]["min_images_per_id"],
        )
        if len(val_dataset) > 0:
            val_loader = DataLoader(
                val_dataset,
                batch_size=cfg["training"].get("val_batch_size", cfg["training"]["batch_size"]),
                shuffle=False,
                num_workers=cfg["training"].get("num_workers", 4),
                pin_memory=torch.backends.mps.is_available(),
            )
            logger.info(f"✅ Val loader: {len(val_dataset)} images")
    
    return train_loader, val_loader