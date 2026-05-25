import torch
from torch.utils.data import DataLoader, random_split
from torchvision import datasets, transforms
from src.augmentation.augmix import get_augmix_transform

def get_cifar10_loaders(
    batch_size=128,
    val_size=5000,
    seed=42,
    num_workers=2,
    augmentation="standard"
):
    """
    Creates CIFAR-10 train, validation, and test loaders.

    CIFAR-10 original training set has 50,000 images.
    We split it into:

        train: 45,000 images
        validation: 5,000 images

    The original CIFAR-10 test set remains untouched.
    """

    if augmentation == "standard":
        train_transform = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.ToTensor(),
        ])

    elif augmentation == "augmix":
        train_transform = get_augmix_transform()

    else:
        raise ValueError(f"Unsupported augmentation: {augmentation}")

    eval_transform = transforms.Compose([
        transforms.ToTensor(),
    ])

    full_train_dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=True,
        transform=train_transform
    )

    test_dataset = datasets.CIFAR10(
        root="./data",
        train=False,
        download=True,
        transform=eval_transform
    )

    train_size = len(full_train_dataset) - val_size

    generator = torch.Generator().manual_seed(seed)

    train_dataset, val_dataset = random_split(
        full_train_dataset,
        [train_size, val_size],
        generator=generator
    )

    # Important:
    # validation should NOT use random crop / flip.
    val_dataset.dataset = datasets.CIFAR10(
        root="./data",
        train=True,
        download=False,
        transform=eval_transform
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=num_workers
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return train_loader, val_loader, test_loader