import argparse

import torch
import torch.nn as nn
from tqdm import tqdm

from src.datasets.cifar10 import get_cifar10_loaders
from src.models.simple_cnn import SimpleCNN
from src.models.resnet18_cifar import ResNet18CIFAR
from src.evaluation.metrics import accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate trained models on clean CIFAR-10 test set"
    )

    parser.add_argument(
        "--model",
        type=str,
        required=True,
        choices=["cnn", "resnet18"]
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    return parser.parse_args()


def build_model(model_name):
    if model_name == "cnn":
        return SimpleCNN(num_classes=10)

    if model_name == "resnet18":
        return ResNet18CIFAR(num_classes=10)

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate(model, loader, criterion, device):
    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Testing"):
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
    print(f"Model: {args.model}")

    _, _, test_loader = get_cifar10_loaders(
        batch_size=args.batch_size
    )

    model = build_model(args.model).to(device)

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=device
    )

    model.load_state_dict(checkpoint["model_state_dict"])

    criterion = nn.CrossEntropyLoss()

    test_loss, test_acc = evaluate(
        model=model,
        loader=test_loader,
        criterion=criterion,
        device=device
    )

    print("\nClean Test Results")
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.4f}")


if __name__ == "__main__":
    main()