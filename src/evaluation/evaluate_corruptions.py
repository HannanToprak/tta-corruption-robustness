import argparse
import json
import os

import torch
import torch.nn as nn
from tqdm import tqdm

from src.datasets.cifar10c import get_cifar10c_loader
from src.models.simple_cnn import SimpleCNN
from src.models.resnet18_cifar import ResNet18CIFAR
from src.evaluation.metrics import accuracy
from src.utils.checkpoint import load_checkpoint


CORRUPTIONS = [
    "gaussian_noise",
    "shot_noise",
    "impulse_noise",
    "defocus_blur",
    "glass_blur",
    "motion_blur",
    "zoom_blur",
    "snow",
    "frost",
    "fog",
    "brightness",
    "contrast",
    "elastic_transform",
    "pixelate",
    "jpeg_compression"
]


def parse_args():
    parser = argparse.ArgumentParser(
        description="Evaluate models on CIFAR-10-C"
    )

    parser.add_argument(
        "--model",
        type=str,
        default="resnet18",
        choices=["cnn", "resnet18"]
    )
    parser.add_argument(
    "--training_type",
    type=str,
    default="standard",
    choices=["standard", "augmix","augmix_full"]
    )

    parser.add_argument(
        "--checkpoint_path",
        type=str,
        required=True
    )

    parser.add_argument(
        "--severity",
        type=int,
        default=1
    )

    parser.add_argument(
        "--batch_size",
        type=int,
        default=128
    )

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/tta_project/results"
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
        for images, labels in tqdm(loader, desc="Evaluating"):
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
    print(f"Training type: {args.training_type}")
    print(f"Severity: {args.severity}")

    model = build_model(args.model).to(device)

    model, checkpoint = load_checkpoint(
        checkpoint_path=args.checkpoint_path,
        model=model,
        device=device
    )

    criterion = nn.CrossEntropyLoss()

    results = {}

    for corruption in CORRUPTIONS:
        print(f"\nEvaluating corruption: {corruption}")

        loader = get_cifar10c_loader(
            corruption=corruption,
            severity=args.severity,
            batch_size=args.batch_size
        )

        loss, acc = evaluate(
            model=model,
            loader=loader,
            criterion=criterion,
            device=device
        )

        results[corruption] = {
            "loss": loss,
            "accuracy": acc
        }

        print(
            f"{corruption} | "
            f"Loss: {loss:.4f} | "
            f"Accuracy: {acc:.4f}"
        )

    mean_accuracy = (
        sum(r["accuracy"] for r in results.values())
        / len(results)
    )

    results["mean_accuracy"] = mean_accuracy

    print(f"\nMean Corruption Accuracy: {mean_accuracy:.4f}")

    os.makedirs(args.output_dir, exist_ok=True)

    output_path = os.path.join( args.output_dir,f"{args.model}_{args.training_type}_severity_{args.severity}_results.json")

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()