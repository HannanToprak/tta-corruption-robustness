import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


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
    "jpeg_compression",
]


RESULT_ROOT = Path("/content/drive/MyDrive/tta_project/results")
TENT_DIR = RESULT_ROOT / "tent"
OUTPUT_DIR = RESULT_ROOT / "collapse_analysis"

MODELS = ["cnn", "resnet18"]
TRAINING_TYPES = ["standard", "augmix", "augmix_full"]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_continual_file(model, training_type, severity):
    candidates = [
        TENT_DIR / f"{model}_{training_type}_tent_continual_severity_{severity}.json",
    ]

    if training_type == "standard":
        candidates += [
            TENT_DIR / f"{model}_tent_continual_severity_{severity}.json",
        ]

    for path in candidates:
        if path.exists():
            return path

    return None


def collect_continual_results():
    rows = []

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            for severity in [1, 2, 3, 4, 5]:
                path = find_continual_file(
                    model=model,
                    training_type=training_type,
                    severity=severity,
                )

                if path is None:
                    continue

                data = load_json(path)

                for order_index, corruption in enumerate(CORRUPTIONS):
                    rows.append({
                        "model": model,
                        "training_type": training_type,
                        "severity": severity,
                        "order_index": order_index,
                        "corruption": corruption,
                        "accuracy": data[corruption]["accuracy"],
                        "entropy": data[corruption]["entropy"],
                        "path": str(path),
                    })

    return pd.DataFrame(rows)


def plot_accuracy_entropy_trajectory(df, model, training_type, severity):
    subset = df[
        (df["model"] == model)
        & (df["training_type"] == training_type)
        & (df["severity"] == severity)
    ].sort_values("order_index")

    if len(subset) == 0:
        return

    fig, ax1 = plt.subplots(figsize=(12, 6))

    ax1.plot(
        subset["order_index"],
        subset["accuracy"],
        marker="o",
        label="Accuracy",
    )
    ax1.set_xlabel("Corruption order")
    ax1.set_ylabel("Accuracy")
    ax1.set_ylim(0.0, 1.0)
    ax1.grid(True, alpha=0.3)

    ax2 = ax1.twinx()
    ax2.plot(
        subset["order_index"],
        subset["entropy"],
        marker="s",
        linestyle="--",
        label="Entropy",
    )
    ax2.set_ylabel("Entropy")

    ax1.set_xticks(range(len(CORRUPTIONS)))
    ax1.set_xticklabels(CORRUPTIONS, rotation=45, ha="right")

    plt.title(
        f"Continual TENT Collapse Trajectory\n"
        f"{model} | {training_type} | severity {severity}"
    )

    lines_1, labels_1 = ax1.get_legend_handles_labels()
    lines_2, labels_2 = ax2.get_legend_handles_labels()
    ax1.legend(lines_1 + lines_2, labels_1 + labels_2, loc="upper right")

    plt.tight_layout()

    path = OUTPUT_DIR / f"{model}_{training_type}_severity_{severity}_accuracy_entropy.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


def plot_cnn_training_comparison(df, severity):
    plt.figure(figsize=(12, 6))

    for training_type in TRAINING_TYPES:
        subset = df[
            (df["model"] == "cnn")
            & (df["training_type"] == training_type)
            & (df["severity"] == severity)
        ].sort_values("order_index")

        if len(subset) == 0:
            continue

        plt.plot(
            subset["order_index"],
            subset["accuracy"],
            marker="o",
            label=training_type,
        )

    plt.xlabel("Corruption order")
    plt.ylabel("Accuracy")
    plt.title(f"CNN Continual TENT Accuracy Collapse | Severity {severity}")
    plt.xticks(range(len(CORRUPTIONS)), CORRUPTIONS, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / f"cnn_training_comparison_severity_{severity}.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


def plot_entropy_accuracy_scatter(df):
    plt.figure(figsize=(8, 6))

    for model in MODELS:
        subset = df[df["model"] == model]

        if len(subset) == 0:
            continue

        plt.scatter(
            subset["entropy"],
            subset["accuracy"],
            alpha=0.6,
            label=model,
        )

    plt.xlabel("Entropy")
    plt.ylabel("Accuracy")
    plt.title("Continual TENT: Entropy vs Accuracy")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend()
    plt.tight_layout()

    path = OUTPUT_DIR / "continual_entropy_vs_accuracy.png"
    plt.savefig(path, dpi=300)
    plt.close()

    print(f"Saved: {path}")


def compute_collapse_metrics(df):
    rows = []

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            for severity in [1, 2, 3, 4, 5]:
                subset = df[
                    (df["model"] == model)
                    & (df["training_type"] == training_type)
                    & (df["severity"] == severity)
                ].sort_values("order_index")

                if len(subset) == 0:
                    continue

                first_acc = subset.iloc[0]["accuracy"]
                last_acc = subset.iloc[-1]["accuracy"]
                min_acc = subset["accuracy"].min()
                mean_acc = subset["accuracy"].mean()

                first_entropy = subset.iloc[0]["entropy"]
                last_entropy = subset.iloc[-1]["entropy"]
                min_entropy = subset["entropy"].min()
                mean_entropy = subset["entropy"].mean()

                rows.append({
                    "model": model,
                    "training_type": training_type,
                    "severity": severity,
                    "first_accuracy": first_acc,
                    "last_accuracy": last_acc,
                    "min_accuracy": min_acc,
                    "mean_accuracy": mean_acc,
                    "accuracy_drop_first_to_last": first_acc - last_acc,
                    "first_entropy": first_entropy,
                    "last_entropy": last_entropy,
                    "min_entropy": min_entropy,
                    "mean_entropy": mean_entropy,
                    "entropy_drop_first_to_last": first_entropy - last_entropy,
                })

    return pd.DataFrame(rows)


def print_key_findings(metrics_df):
    print("\nCollapse Metrics")
    print(
        metrics_df
        .sort_values(["model", "training_type", "severity"])
        .to_string(index=False)
    )

    print("\nMost severe accuracy drops")
    print(
        metrics_df
        .sort_values("accuracy_drop_first_to_last", ascending=False)
        .head(10)
        .to_string(index=False)
    )

    print("\nStrong confident-collapse cases")
    confident_collapse = metrics_df[
        (metrics_df["last_accuracy"] < 0.25)
        & (metrics_df["last_entropy"] < metrics_df["first_entropy"])
    ]

    if len(confident_collapse) == 0:
        print("No strong confident-collapse cases found.")
    else:
        print(
            confident_collapse
            .sort_values(["model", "training_type", "severity"])
            .to_string(index=False)
        )


def main():
    os.makedirs(OUTPUT_DIR, exist_ok=True)

    df = collect_continual_results()

    if len(df) == 0:
        raise RuntimeError("No continual TENT result files found.")

    results_csv = OUTPUT_DIR / "continual_collapse_results.csv"
    df.to_csv(results_csv, index=False)
    print(f"Saved: {results_csv}")

    metrics_df = compute_collapse_metrics(df)

    metrics_csv = OUTPUT_DIR / "collapse_metrics.csv"
    metrics_df.to_csv(metrics_csv, index=False)
    print(f"Saved: {metrics_csv}")

    print_key_findings(metrics_df)

    for severity in [1, 2, 3, 4, 5]:
        plot_cnn_training_comparison(df, severity=severity)

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            for severity in [1, 3, 5]:
                plot_accuracy_entropy_trajectory(
                    df=df,
                    model=model,
                    training_type=training_type,
                    severity=severity,
                )

    plot_entropy_accuracy_scatter(df)


if __name__ == "__main__":
    main()