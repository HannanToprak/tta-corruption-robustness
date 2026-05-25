import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

RESULT_ROOT = Path("/content/drive/MyDrive/tta_project/results")
FROZEN_DIR = RESULT_ROOT / "standard_frozen"
TENT_DIR = RESULT_ROOT / "tent"
CLEAN_DIR = RESULT_ROOT / "clean"
PLOT_DIR = RESULT_ROOT / "plots"

MODELS = ["cnn", "resnet18"]
TRAINING_TYPES = ["standard", "augmix","augmix_full"]
METHODS = ["frozen", "episodic_tent", "continual_tent"]


def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def find_result_file(model, training_type, severity, method):
    if method == "frozen":
        candidates = [
            FROZEN_DIR / f"{model}_{training_type}_severity_{severity}_results.json",
            RESULT_ROOT / f"{model}_{training_type}_severity_{severity}_results.json",
        ]

        if training_type == "standard":
            candidates += [
                FROZEN_DIR / f"{model}_severity_{severity}_results.json",
                RESULT_ROOT / f"{model}_severity_{severity}_results.json",
            ]

    elif method == "episodic_tent":
        candidates = [
            TENT_DIR / f"{model}_{training_type}_tent_episodic_severity_{severity}.json",
        ]

        if training_type == "standard":
            candidates += [
                TENT_DIR / f"{model}_tent_episodic_severity_{severity}.json",
                TENT_DIR / f"{model}_tent_severity_{severity}.json",
            ]

    elif method == "continual_tent":
        candidates = [
            TENT_DIR / f"{model}_{training_type}_tent_continual_severity_{severity}.json",
        ]

        if training_type == "standard":
            candidates += [
                TENT_DIR / f"{model}_tent_continual_severity_{severity}.json",
            ]

    else:
        raise ValueError(f"Unsupported method: {method}")

    for path in candidates:
        if path.exists():
            return path

    return None


def collect_mean_results():
    rows = []

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            for severity in [1, 2, 3, 4, 5]:
                for method in METHODS:
                    path = find_result_file(
                        model=model,
                        training_type=training_type,
                        severity=severity,
                        method=method,
                    )

                    if path is None:
                        continue

                    data = load_json(path)

                    rows.append({
                        "model": model,
                        "training_type": training_type,
                        "method": method,
                        "severity": severity,
                        "mean_accuracy": data["mean_accuracy"],
                        "path": str(path),
                    })

    return pd.DataFrame(rows)


def collect_corruption_results():
    rows = []

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            for severity in [1, 2, 3, 4, 5]:
                for method in METHODS:
                    path = find_result_file(
                        model=model,
                        training_type=training_type,
                        severity=severity,
                        method=method,
                    )

                    if path is None:
                        continue

                    data = load_json(path)

                    for corruption in CORRUPTIONS:
                        rows.append({
                            "model": model,
                            "training_type": training_type,
                            "method": method,
                            "severity": severity,
                            "corruption": corruption,
                            "accuracy": data[corruption]["accuracy"],
                            "entropy": data[corruption].get("entropy"),
                            "path": str(path),
                        })

    return pd.DataFrame(rows)


def plot_severity_vs_accuracy(mean_df):
    plt.figure(figsize=(12, 7))

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            for method in METHODS:
                subset = mean_df[
                    (mean_df["model"] == model)
                    & (mean_df["training_type"] == training_type)
                    & (mean_df["method"] == method)
                ].sort_values("severity")

                if len(subset) == 0:
                    continue

                plt.plot(
                    subset["severity"],
                    subset["mean_accuracy"],
                    marker="o",
                    label=f"{model} | {training_type} | {method}",
                )

    plt.xlabel("Corruption severity")
    plt.ylabel("Mean corruption accuracy")
    plt.title("Mean Accuracy across Corruption Severity")
    plt.xticks([1, 2, 3, 4, 5])
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = PLOT_DIR / "severity_vs_accuracy_all.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_model_training_comparison(mean_df, model):
    plt.figure(figsize=(10, 6))

    for training_type in TRAINING_TYPES:
        for method in METHODS:
            subset = mean_df[
                (mean_df["model"] == model)
                & (mean_df["training_type"] == training_type)
                & (mean_df["method"] == method)
            ].sort_values("severity")

            if len(subset) == 0:
                continue

            plt.plot(
                subset["severity"],
                subset["mean_accuracy"],
                marker="o",
                label=f"{training_type} | {method}",
            )

    plt.xlabel("Corruption severity")
    plt.ylabel("Mean corruption accuracy")
    plt.title(f"{model}: Standard vs AugMix across Severity")
    plt.xticks([1, 2, 3, 4, 5])
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    path = PLOT_DIR / f"{model}_standard_vs_augmix.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_tent_gain(mean_df):
    plt.figure(figsize=(10, 6))

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            frozen = mean_df[
                (mean_df["model"] == model)
                & (mean_df["training_type"] == training_type)
                & (mean_df["method"] == "frozen")
            ][["severity", "mean_accuracy"]].rename(
                columns={"mean_accuracy": "frozen_accuracy"}
            )

            for method in ["episodic_tent", "continual_tent"]:
                tent = mean_df[
                    (mean_df["model"] == model)
                    & (mean_df["training_type"] == training_type)
                    & (mean_df["method"] == method)
                ][["severity", "mean_accuracy"]].rename(
                    columns={"mean_accuracy": "tent_accuracy"}
                )

                merged = pd.merge(frozen, tent, on="severity", how="inner")

                if len(merged) == 0:
                    continue

                merged["gain"] = merged["tent_accuracy"] - merged["frozen_accuracy"]

                plt.plot(
                    merged["severity"],
                    merged["gain"],
                    marker="o",
                    label=f"{model} | {training_type} | {method}",
                )

    plt.axhline(0.0, linestyle="--")
    plt.xlabel("Corruption severity")
    plt.ylabel("Accuracy gain over frozen")
    plt.title("TENT Gain over Frozen Baseline")
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=8)
    plt.tight_layout()

    path = PLOT_DIR / "tent_gain_over_frozen_all.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_continual_trajectory(corr_df, severity):
    plt.figure(figsize=(12, 6))

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            subset = corr_df[
                (corr_df["model"] == model)
                & (corr_df["training_type"] == training_type)
                & (corr_df["method"] == "continual_tent")
                & (corr_df["severity"] == severity)
            ]

            if len(subset) == 0:
                continue

            subset = subset.set_index("corruption").loc[CORRUPTIONS].reset_index()

            plt.plot(
                range(len(CORRUPTIONS)),
                subset["accuracy"],
                marker="o",
                label=f"{model} | {training_type}",
            )

    plt.xlabel("Corruption order")
    plt.ylabel("Accuracy")
    plt.title(f"Continual TENT Trajectory at Severity {severity}")
    plt.xticks(range(len(CORRUPTIONS)), CORRUPTIONS, rotation=45, ha="right")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    path = PLOT_DIR / f"continual_trajectory_severity_{severity}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_heatmap(corr_df, model, training_type, severity):
    methods = METHODS
    rows = []

    for corruption in CORRUPTIONS:
        row = []

        for method in methods:
            values = corr_df[
                (corr_df["model"] == model)
                & (corr_df["training_type"] == training_type)
                & (corr_df["method"] == method)
                & (corr_df["severity"] == severity)
                & (corr_df["corruption"] == corruption)
            ]["accuracy"]

            row.append(float("nan") if len(values) == 0 else values.iloc[0])

        rows.append(row)

    heatmap_df = pd.DataFrame(rows, index=CORRUPTIONS, columns=methods)

    plt.figure(figsize=(7, 9))
    plt.imshow(heatmap_df.values, aspect="auto", vmin=0.0, vmax=1.0)
    plt.colorbar(label="Accuracy")
    plt.xticks(range(len(methods)), methods, rotation=30, ha="right")
    plt.yticks(range(len(CORRUPTIONS)), CORRUPTIONS)
    plt.title(f"{model} | {training_type} | Severity {severity}")

    for i in range(len(CORRUPTIONS)):
        for j in range(len(methods)):
            value = heatmap_df.iloc[i, j]
            if pd.notna(value):
                plt.text(j, i, f"{value:.2f}", ha="center", va="center", fontsize=8)

    plt.tight_layout()

    path = PLOT_DIR / f"{model}_{training_type}_heatmap_severity_{severity}.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def plot_entropy_vs_accuracy(corr_df):
    subset = corr_df[
        corr_df["method"].isin(["episodic_tent", "continual_tent"])
        & corr_df["entropy"].notna()
    ]

    plt.figure(figsize=(9, 6))

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            model_subset = subset[
                (subset["model"] == model)
                & (subset["training_type"] == training_type)
            ]

            if len(model_subset) == 0:
                continue

            plt.scatter(
                model_subset["entropy"],
                model_subset["accuracy"],
                label=f"{model} | {training_type}",
                alpha=0.7,
            )

    plt.xlabel("Entropy")
    plt.ylabel("Accuracy")
    plt.title("Entropy vs Accuracy under TENT")
    plt.ylim(0.0, 1.0)
    plt.grid(True, alpha=0.3)
    plt.legend(fontsize=9)
    plt.tight_layout()

    path = PLOT_DIR / "entropy_vs_accuracy_all.png"
    plt.savefig(path, dpi=300)
    plt.close()
    print(f"Saved: {path}")


def print_summary(mean_df):
    print("\nMean Results")
    print(
        mean_df
        .sort_values(["model", "training_type", "method", "severity"])
        [["model", "training_type", "method", "severity", "mean_accuracy"]]
        .to_string(index=False)
    )


def main():
    os.makedirs(PLOT_DIR, exist_ok=True)

    mean_df = collect_mean_results()
    corr_df = collect_corruption_results()

    if len(mean_df) == 0:
        raise RuntimeError("No result files found. Check result paths.")

    mean_df.to_csv(PLOT_DIR / "mean_results.csv", index=False)
    corr_df.to_csv(PLOT_DIR / "corruption_results.csv", index=False)

    print_summary(mean_df)

    plot_severity_vs_accuracy(mean_df)

    for model in MODELS:
        plot_model_training_comparison(mean_df, model=model)

    plot_tent_gain(mean_df)

    for severity in [1, 2, 3, 4, 5]:
        plot_continual_trajectory(corr_df, severity=severity)

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            for severity in [1, 5]:
                plot_heatmap(
                    corr_df,
                    model=model,
                    training_type=training_type,
                    severity=severity,
                )

    plot_entropy_vs_accuracy(corr_df)


if __name__ == "__main__":
    main()