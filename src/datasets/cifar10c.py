import os

import numpy as np
from PIL import Image
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms


class CIFAR10CDataset(Dataset):
    """
    CIFAR-10-C dataset loader.

    CIFAR-10-C stores each corruption as a .npy file with 50,000 images.
    These 50,000 images are divided into 5 severity levels:

        severity 1 -> images 0:10000
        severity 2 -> images 10000:20000
        severity 3 -> images 20000:30000
        severity 4 -> images 30000:40000
        severity 5 -> images 40000:50000
    """

    def __init__(self, root, corruption, severity, transform=None):
        """
        Args:
            root: path to CIFAR-10-C folder
            corruption: corruption name, e.g. 'gaussian_noise'
            severity: integer from 1 to 5
            transform: torchvision transforms
        """

        if severity < 1 or severity > 5:
            raise ValueError("severity must be between 1 and 5")

        self.root = root
        self.corruption = corruption
        self.severity = severity
        self.transform = transform

        data_path = os.path.join(root, f"{corruption}.npy")
        label_path = os.path.join(root, "labels.npy")

        if not os.path.exists(data_path):
            raise FileNotFoundError(f"Corruption file not found: {data_path}")

        if not os.path.exists(label_path):
            raise FileNotFoundError(f"Labels file not found: {label_path}")

        images = np.load(data_path)
        labels = np.load(label_path)

        start = (severity - 1) * 10000
        end = severity * 10000

        self.images = images[start:end]
        self.labels = labels[start:end]

    def __len__(self):
        return len(self.images)

    def __getitem__(self, index):
        image = self.images[index]
        label = int(self.labels[index])

        image = Image.fromarray(image)

        if self.transform is not None:
            image = self.transform(image)

        return image, label


def get_cifar10c_loader(
    root="/content/drive/MyDrive/tta_project/data/CIFAR-10-C",
    corruption="gaussian_noise",
    severity=1,
    batch_size=128,
    num_workers=2
):
    """
    Creates a DataLoader for a selected CIFAR-10-C corruption and severity.
    """

    transform = transforms.Compose([
        transforms.ToTensor()
    ])

    dataset = CIFAR10CDataset(
        root=root,
        corruption=corruption,
        severity=severity,
        transform=transform
    )

    loader = DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=num_workers
    )

    return loader