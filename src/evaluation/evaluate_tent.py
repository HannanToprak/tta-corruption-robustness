import argparse
import json
import os

import torch
from tqdm import tqdm

from src.datasets.cifar10c import get_cifar10c_loader
from src.models.simple_cnn import SimpleCNN
from src.models.resnet18_cifar import ResNet18CIFAR
from src.evaluation.metrics import accuracy
from src.adaptation.tent import Tent


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
        description="Evaluate TENT on CIFAR-10-C"
    )

    parser.add_argument("--model", type=str, required=True, choices=["cnn", "resnet18"])
    parser.add_argument("--checkpoint_path", type=str, required=True)
    parser.add_argument("--severity", type=int, default=1)
    parser.add_argument("--batch_size", type=int, default=128)
    parser.add_argument("--lr", type=float, default=1e-3)

    parser.add_argument(
        "--mode",
        type=str,
        default="episodic",
        choices=["episodic", "continual"],
        help="episodic resets model for each corruption; continual adapts across corruptions"
    )
    parser.add_argument(
    "--training_type",
    type=str,
    default="standard",
    choices=["standard", "augmix","augmix_full"])

    parser.add_argument(
        "--output_dir",
        type=str,
        default="/content/drive/MyDrive/tta_project/results/tent"
    )

    return parser.parse_args()


def build_model(model_name):
    if model_name == "cnn":
        return SimpleCNN(num_classes=10)

    if model_name == "resnet18":
        return ResNet18CIFAR(num_classes=10)

    raise ValueError(f"Unsupported model: {model_name}")


def evaluate_tent(tent_model, loader, device):
    running_acc = 0.0
    running_entropy = 0.0

    for images, labels in tqdm(loader, desc="TENT Evaluation"):
        images = images.to(device)
        labels = labels.to(device)

        outputs, entropy_loss = tent_model.forward_and_adapt(images)

        running_acc += accuracy(outputs, labels)
        running_entropy += entropy_loss.item()

    avg_acc = running_acc / len(loader)
    avg_entropy = running_entropy / len(loader)

    return avg_acc, avg_entropy


def main():
    args = parse_args()

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    print(f"Using device: {device}")
    print(f"Model: {args.model}")
    print(f"Training type: {args.training_type}")
    print(f"Severity: {args.severity}")
    print(f"TENT mode: {args.mode}")

    os.makedirs(args.output_dir, exist_ok=True)

    checkpoint = torch.load(
        args.checkpoint_path,
        map_location=device
    )

    results = {}

    if args.mode == "continual":
        model = build_model(args.model).to(device)
        model.load_state_dict(checkpoint["model_state_dict"])

        tent_model = Tent(
            model=model,
            lr=args.lr
        )

    for corruption in CORRUPTIONS:
        print(f"\nEvaluating corruption with TENT: {corruption}")

        if args.mode == "episodic":
            model = build_model(args.model).to(device)
            model.load_state_dict(checkpoint["model_state_dict"])

            tent_model = Tent(
                model=model,
                lr=args.lr
            )

        loader = get_cifar10c_loader(
            corruption=corruption,
            severity=args.severity,
            batch_size=args.batch_size
        )

        acc, entropy = evaluate_tent(
            tent_model=tent_model,
            loader=loader,
            device=device
        )

        results[corruption] = {
            "accuracy": acc,
            "entropy": entropy
        }

        print(
            f"{corruption} | "
            f"Accuracy: {acc:.4f} | "
            f"Entropy: {entropy:.4f}"
        )

    mean_accuracy = (
        sum(r["accuracy"] for r in results.values())
        / len(results)
    )

    results["mean_accuracy"] = mean_accuracy
    results["mode"] = args.mode
    results["severity"] = args.severity
    results["model"] = args.model

    print(f"\nMean TENT Accuracy: {mean_accuracy:.4f}")

    output_path = os.path.join(
        args.output_dir,
        f"{args.model}_{args.training_type}_tent_{args.mode}_severity_{args.severity}.json"
    )

    with open(output_path, "w") as f:
        json.dump(results, f, indent=4)

    print(f"Results saved to {output_path}")


if __name__ == "__main__":
    main()