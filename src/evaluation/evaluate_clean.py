import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from src.datasets.cifar10 import get_cifar10_loaders
from src.models.simple_cnn import SimpleCNN
from src.evaluation.metrics import accuracy
from src.utils.checkpoint import load_checkpoint


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate SimpleCNN on clean CIFAR-10 test set"
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        default="/content/drive/MyDrive/tta_project/checkpoints/best_simple_cnn.pth"
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    return parser.parse_args()


def evaluate(model, loader, criterion, device):
    """
    Evaluates model on clean CIFAR-10 test data.
    """

    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Clean Evaluation"):
            images = images.to(device)
            labels = labels.to(device)

            outputs = model(images)
            loss = criterion(outputs, labels)

            running_loss += loss.item()
            running_acc += accuracy(outputs, labels)

    avg_loss = running_loss / len(loader)
    avg_acc = running_acc / len(loader)

    return avg_loss, avg_acc


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")

    _, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size
    )

    model = SimpleCNN(num_classes=10).to(device)

    model, checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model=model,
        device=device
    )

    print(f"Loaded checkpoint from epoch: {checkpoint['epoch']}")
    print(f"Best recorded test accuracy: {checkpoint['best_test_acc']:.4f}")

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device
    )

    print("\nClean CIFAR-10 Evaluation")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()