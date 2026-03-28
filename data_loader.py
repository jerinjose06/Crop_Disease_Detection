# data_loader.py

import os
import random
import shutil
import numpy as np
from PIL import Image

import torch
from torch.utils.data import Dataset, DataLoader, Subset
from torchvision import transforms

import config


def set_seed(seed=config.SEED):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False


class PlantDiseaseDataset(Dataset):
    """Custom Dataset for Plant Disease Images."""

    def __init__(self, root_dir, transform=None):
        """
        Args:
            root_dir (str): Directory with all the class folders.
            transform: Optional transform to be applied on images.
        """
        self.root_dir = root_dir
        self.transform = transform
        self.images = []
        self.labels = []
        self.class_to_idx = {}
        self.idx_to_class = {}

        # Load all image paths and labels
        self._load_dataset()

    def _load_dataset(self):
        """Load dataset from directory structure."""
        classes = sorted(os.listdir(self.root_dir))
        classes = [c for c in classes if os.path.isdir(os.path.join(self.root_dir, c))]

        for idx, class_name in enumerate(classes):
            self.class_to_idx[class_name] = idx
            self.idx_to_class[idx] = class_name

            class_dir = os.path.join(self.root_dir, class_name)
            for img_name in os.listdir(class_dir):
                img_path = os.path.join(class_dir, img_name)
                if img_path.lower().endswith(('.png', '.jpg', '.jpeg', '.bmp', '.gif')):
                    self.images.append(img_path)
                    self.labels.append(idx)

        print(f"Loaded {len(self.images)} images from {len(classes)} classes in {self.root_dir}")

    def __len__(self):
        return len(self.images)

    def __getitem__(self, idx):
        """Get image and label at index."""
        img_path = self.images[idx]
        label = self.labels[idx]

        # Load image
        try:
            image = Image.open(img_path).convert('RGB')
        except Exception as e:
            print(f"Error loading image {img_path}: {e}")
            # Return a blank image if loading fails
            image = Image.new('RGB', config.IMAGE_SIZE, (0, 0, 0))

        if self.transform:
            image = self.transform(image)

        return image, label


def get_transforms():
    """Define data transforms for training, validation, and testing."""

    # Training transforms with data augmentation
    train_transform = transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomCrop(config.IMAGE_SIZE),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomVerticalFlip(p=0.3),
        transforms.RandomRotation(degrees=30),
        transforms.ColorJitter(
            brightness=0.2,
            contrast=0.2,
            saturation=0.2,
            hue=0.1
        ),
        transforms.RandomAffine(
            degrees=0,
            translate=(0.1, 0.1),
            scale=(0.9, 1.1)
        ),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    # Validation and Test transforms (no augmentation)
    eval_transform = transforms.Compose([
        transforms.Resize(config.IMAGE_SIZE),
        transforms.ToTensor(),
        transforms.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
    ])

    return train_transform, eval_transform


def create_test_split(dataset, test_ratio=config.TEST_SPLIT_RATIO):
    """
    Split training dataset into train and test subsets.
    Performs stratified split to maintain class distribution.
    """
    set_seed()

    labels = np.array(dataset.labels)
    num_classes = len(dataset.class_to_idx)

    train_indices = []
    test_indices = []

    for class_idx in range(num_classes):
        class_indices = np.where(labels == class_idx)[0].tolist()
        random.shuffle(class_indices)

        n_test = max(1, int(len(class_indices) * test_ratio))
        test_indices.extend(class_indices[:n_test])
        train_indices.extend(class_indices[n_test:])

    random.shuffle(train_indices)
    random.shuffle(test_indices)

    return train_indices, test_indices


def get_data_loaders():
    """Create DataLoaders for training, validation, and testing."""
    set_seed()

    train_transform, eval_transform = get_transforms()

    # Load full training dataset with train transform
    full_train_dataset = PlantDiseaseDataset(
        root_dir=config.TRAIN_DIR,
        transform=train_transform
    )

    # Create a copy with eval transform for test split
    full_train_dataset_eval = PlantDiseaseDataset(
        root_dir=config.TRAIN_DIR,
        transform=eval_transform
    )

    # Split training data into train and test
    train_indices, test_indices = create_test_split(full_train_dataset)

    # Create subsets
    train_dataset = Subset(full_train_dataset, train_indices)
    test_dataset = Subset(full_train_dataset_eval, test_indices)

    # Load validation dataset
    valid_dataset = PlantDiseaseDataset(
        root_dir=config.VALID_DIR,
        transform=eval_transform
    )

    print(f"\n{'='*50}")
    print(f"Dataset Split Summary:")
    print(f"{'='*50}")
    print(f"Training samples:   {len(train_dataset)}")
    print(f"Validation samples: {len(valid_dataset)}")
    print(f"Testing samples:    {len(test_dataset)}")
    print(f"Number of classes:  {len(full_train_dataset.class_to_idx)}")
    print(f"{'='*50}\n")

    # Create DataLoaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=True,
        num_workers=4,
        pin_memory=True,
        drop_last=True
    )

    valid_loader = DataLoader(
        valid_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=config.BATCH_SIZE,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )

    # Store class mappings
    class_to_idx = full_train_dataset.class_to_idx
    idx_to_class = full_train_dataset.idx_to_class

    return train_loader, valid_loader, test_loader, class_to_idx, idx_to_class


def get_class_distribution(dataset, indices=None):
    """Get the distribution of classes in a dataset."""
    if indices is not None:
        labels = [dataset.labels[i] for i in indices]
    else:
        labels = dataset.labels

    distribution = {}
    for label in labels:
        class_name = dataset.idx_to_class[label]
        distribution[class_name] = distribution.get(class_name, 0) + 1

    return distribution