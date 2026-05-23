import os
import json
import argparse

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm

from src.datasets.cifar10 import get_cifar10_loaders
from src.models.resnet18_cifar import ResNet18CIFAR
from src.models.simple_cnn import SimpleCNN
from src.evaluation.metrics import accuracy


def parse_args():
    parser = argparse.ArgumentParser(
        description="Train image classification models on CIFAR-10"
    )

    parser.add_argument("--model", type=str, default="cnn", choices=["cnn", "resnet18"])
    parser.add_argument("--epochs", type=int, default=30)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)
    parser.add_argument("--weight_decay", type=float, default=1e-4)

    parser.add_argument(
        "--checkpoint_dir",
        type=str,
        default="/content/drive/MyDrive/tta_project/checkpoints"
    )

    parser.add_argument(
        "--history_dir",
        type=str,
        default="/content/drive/MyDrive/tta_project/results"
    )

    parser.add_argument("--checkpoint_name", type=str, default=None)
    parser.add_argument("--history_name", type=str, default=None)

    return parser.parse_args()


def build_model(model_name, num_classes=10):
    """
    Builds the selected model architecture.
    """

    if model_name == "cnn":
        return SimpleCNN(num_classes=num_classes)

    if model_name == "resnet18":
        return ResNet18CIFAR(num_classes=num_classes)

    raise ValueError(f"Unsupported model: {model_name}")


def train_one_epoch(model, loader, criterion, optimizer, device):
    """
    Trains the model for one epoch.
    """

    model.train()

    running_loss = 0.0
    running_acc = 0.0

    for images, labels in tqdm(loader, desc="Training"):
        images = images.to(device)
        labels = labels.to(device)

        optimizer.zero_grad()

        outputs = model(images)
        loss = criterion(outputs, labels)

        loss.backward()
        optimizer.step()

        running_loss += loss.item()
        running_acc += accuracy(outputs, labels)

    avg_loss = running_loss / len(loader)
    avg_acc = running_acc / len(loader)

    return avg_loss, avg_acc


def evaluate(model, loader, criterion, device):
    """
    Evaluates the model without updating parameters.
    """

    model.eval()

    running_loss = 0.0
    running_acc = 0.0

    with torch.no_grad():
        for images, labels in tqdm(loader, desc="Evaluation"):
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

    if args.checkpoint_name is None:
        args.checkpoint_name = f"best_{args.model}.pth"

    if args.history_name is None:
        args.history_name = f"{args.model}_training_history.json"

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    print(f"Using device: {device}")
    print(f"Selected model: {args.model}")

    os.makedirs(args.checkpoint_dir, exist_ok=True)
    os.makedirs(args.history_dir, exist_ok=True)

    train_loader, val_loader, _= get_cifar10_loaders(
        batch_size=args.batch_size
    )

    model = build_model(
        model_name=args.model,
        num_classes=10
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = optim.Adam(
        model.parameters(),
        lr=args.lr,
        weight_decay=args.weight_decay
    )

    scheduler = optim.lr_scheduler.StepLR(
        optimizer,
        step_size=10,
        gamma=0.5
    )

    best_test_acc = 0.0

    history = {
        "model": args.model,
        "train_loss": [],
        "train_acc": [],
        "test_loss": [],
        "test_acc": [],
        "learning_rate": []
    }

    for epoch in range(args.epochs):
        print(f"\nEpoch {epoch + 1}/{args.epochs}")

        train_loss, train_acc = train_one_epoch(
            model=model,
            loader=train_loader,
            criterion=criterion,
            optimizer=optimizer,
            device=device
        )

        test_loss, test_acc = evaluate(
            model=model,
            loader=test_loader,
            criterion=criterion,
            device=device
        )

        current_lr = optimizer.param_groups[0]["lr"]

        history["train_loss"].append(train_loss)
        history["train_acc"].append(train_acc)
        history["test_loss"].append(test_loss)
        history["test_acc"].append(test_acc)
        history["learning_rate"].append(current_lr)

        print(
            f"Train Loss: {train_loss:.4f} | "
            f"Train Acc: {train_acc:.4f} | "
            f"Test Loss: {test_loss:.4f} | "
            f"Test Acc: {test_acc:.4f} | "
            f"LR: {current_lr:.6f}"
        )

        if test_acc > best_test_acc:
            best_test_acc = test_acc

            checkpoint_path = os.path.join(
                args.checkpoint_dir,
                args.checkpoint_name
            )

            torch.save(
                {
                    "epoch": epoch + 1,
                    "model": args.model,
                    "model_state_dict": model.state_dict(),
                    "optimizer_state_dict": optimizer.state_dict(),
                    "best_test_acc": best_test_acc,
                    "history": history,
                    "args": vars(args)
                },
                checkpoint_path
            )

            print(f"Best model saved with accuracy: {best_test_acc:.4f}")

        scheduler.step()

    history_path = os.path.join(
        args.history_dir,
        args.history_name
    )

    with open(history_path, "w") as f:
        json.dump(history, f, indent=4)

    print("\nTraining completed.")
    print(f"Best test accuracy: {best_test_acc:.4f}")
    print(f"Training history saved to {history_path}")


if __name__ == "__main__":
    main()