import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# LARGE PAPER-READABLE GLOBAL PLOT STYLE
# ============================================================
# If figures become too large, reduce SCALE to 0.85 or 0.75.
SCALE = 1.0

plt.rcParams.update({
    "font.size": int(34 * SCALE),
    "axes.titlesize": int(40 * SCALE),
    "axes.labelsize": int(38 * SCALE),
    "xtick.labelsize": int(30 * SCALE),
    "ytick.labelsize": int(30 * SCALE),
    "legend.fontsize": int(28 * SCALE),
    "figure.titlesize": int(42 * SCALE),
    "lines.linewidth": 7,
    "lines.markersize": 16,
    "axes.linewidth": 2.5,
    "xtick.major.width": 2.2,
    "ytick.major.width": 2.2,
    "xtick.major.size": 9,
    "ytick.major.size": 9,
    "savefig.dpi": 600,
    "figure.dpi": 160,
})


CORRUPTIONS = [
    "gaussian_noise", "shot_noise", "impulse_noise",
    "defocus_blur", "glass_blur", "motion_blur", "zoom_blur",
    "snow", "frost", "fog", "brightness", "contrast",
    "elastic_transform", "pixelate", "jpeg_compression",
]

# Short labels make x-axis much more readable in the paper.
SHORT_CORRUPTION_NAMES = {
    "gaussian_noise": "Gauss.",
    "shot_noise": "Shot",
    "impulse_noise": "Impulse",
    "defocus_blur": "Defocus",
    "glass_blur": "Glass",
    "motion_blur": "Motion",
    "zoom_blur": "Zoom",
    "snow": "Snow",
    "frost": "Frost",
    "fog": "Fog",
    "brightness": "Bright.",
    "contrast": "Contrast",
    "elastic_transform": "Elastic",
    "pixelate": "Pixel",
    "jpeg_compression": "JPEG",
}

MODEL_NAMES = {
    "cnn": "CNN",
    "resnet18": "ResNet18",
}

TRAINING_NAMES = {
    "standard": "Standard",
    "augmix": "Simple AugMix",
    "augmix_full": "Full AugMix",
}

METHOD_NAMES = {
    "frozen": "Frozen",
    "episodic_tent": "Episodic TENT",
    "continual_tent": "Continual TENT",
}


RESULT_ROOT = Path("/content/drive/MyDrive/tta_project/results")
FROZEN_DIR = RESULT_ROOT / "standard_frozen"
TENT_DIR = RESULT_ROOT / "tent"
CLEAN_DIR = RESULT_ROOT / "clean"
PLOT_DIR = RESULT_ROOT / "plots_readable"

MODELS = ["cnn", "resnet18"]
TRAINING_TYPES = ["standard", "augmix", "augmix_full"]
METHODS = ["frozen", "episodic_tent", "continual_tent"]


# ============================================================
# Utility functions
# ============================================================

def load_json(path):
    with open(path, "r") as f:
        return json.load(f)


def save_figure(path):
    """
    Saves each figure as both PNG and PDF.
    PDF is better for papers because it stays sharp when enlarged.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    png_path = path.with_suffix(".png")
    pdf_path = path.with_suffix(".pdf")

    plt.savefig(png_path, dpi=600, bbox_inches="tight", pad_inches=0.18)
    plt.savefig(pdf_path, bbox_inches="tight", pad_inches=0.18)
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def format_accuracy_axis(ax, ymin=0, ymax=100):
    """
    Use percentage scale instead of 0-1 scale.
    """
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.45, linewidth=2.0)
    ax.tick_params(axis="both", labelsize=int(30 * SCALE))


def accuracy_to_percent(series):
    return series * 100.0


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


# ============================================================
# Data collection
# ============================================================

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
                        if corruption not in data:
                            continue

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


# ============================================================
# Plot 1: Mean accuracy across severity
# ============================================================

def plot_severity_vs_accuracy(mean_df):
    plt.figure(figsize=(34, 20))

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

                label = (
                    f"{MODEL_NAMES[model]} | "
                    f"{TRAINING_NAMES[training_type]} | "
                    f"{METHOD_NAMES[method]}"
                )

                plt.plot(
                    subset["severity"],
                    accuracy_to_percent(subset["mean_accuracy"]),
                    marker="o",
                    linewidth=7,
                    markersize=16,
                    label=label,
                )

    ax = plt.gca()
    format_accuracy_axis(ax)

    plt.xlabel("Corruption Severity", fontweight="bold")
    plt.ylabel("Mean Corruption Accuracy (%)", fontweight="bold")
    plt.title("Mean Accuracy across Corruption Severity", fontweight="bold")
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend(fontsize=int(24 * SCALE), ncol=2, frameon=True)
    plt.tight_layout()

    save_figure(PLOT_DIR / "severity_vs_accuracy_all_large")


# ============================================================
# Plot 2: Grid plot by model and training type
# ============================================================

def plot_severity_vs_accuracy_grid(mean_df):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(42, 24),
        sharex=True,
        sharey=True,
    )

    method_styles = {
        "frozen": {
            "linestyle": "-",
            "marker": "o",
            "label": "Frozen",
        },
        "episodic_tent": {
            "linestyle": "--",
            "marker": "s",
            "label": "Episodic TENT",
        },
        "continual_tent": {
            "linestyle": ":",
            "marker": "^",
            "label": "Continual TENT",
        },
    }

    for row, model in enumerate(MODELS):
        for col, training_type in enumerate(TRAINING_TYPES):
            ax = axes[row, col]

            for method in METHODS:
                subset = mean_df[
                    (mean_df["model"] == model)
                    & (mean_df["training_type"] == training_type)
                    & (mean_df["method"] == method)
                ].sort_values("severity")

                if len(subset) == 0:
                    continue

                style = method_styles[method]

                ax.plot(
                    subset["severity"],
                    accuracy_to_percent(subset["mean_accuracy"]),
                    linestyle=style["linestyle"],
                    marker=style["marker"],
                    linewidth=7,
                    markersize=16,
                    label=style["label"],
                )

            ax.set_title(
                TRAINING_NAMES[training_type],
                fontsize=int(38 * SCALE),
                fontweight="bold",
                pad=18,
            )

            ax.set_xticks([1, 2, 3, 4, 5])
            format_accuracy_axis(ax)

            if col == 0:
                ax.set_ylabel(
                    f"{MODEL_NAMES[model]}\nAccuracy (%)",
                    fontsize=int(38 * SCALE),
                    fontweight="bold",
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()

    fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        fontsize=int(32 * SCALE),
        bbox_to_anchor=(0.5, 1.03),
    )

    fig.supxlabel(
        "Corruption Severity",
        fontsize=int(42 * SCALE),
        fontweight="bold",
        y=0.02,
    )

    fig.supylabel(
        "Mean CIFAR-10-C Accuracy (%)",
        fontsize=int(42 * SCALE),
        fontweight="bold",
        x=0.01,
    )

    plt.tight_layout(rect=[0.03, 0.05, 1, 0.94])

    save_figure(PLOT_DIR / "severity_vs_accuracy_grid_large")


# ============================================================
# Plot 3: Model-specific comparison
# ============================================================

def plot_model_training_comparison(mean_df, model):
    plt.figure(figsize=(34, 20))

    for training_type in TRAINING_TYPES:
        for method in METHODS:
            subset = mean_df[
                (mean_df["model"] == model)
                & (mean_df["training_type"] == training_type)
                & (mean_df["method"] == method)
            ].sort_values("severity")

            if len(subset) == 0:
                continue

            label = f"{TRAINING_NAMES[training_type]} | {METHOD_NAMES[method]}"

            plt.plot(
                subset["severity"],
                accuracy_to_percent(subset["mean_accuracy"]),
                marker="o",
                linewidth=7,
                markersize=16,
                label=label,
            )

    ax = plt.gca()
    format_accuracy_axis(ax)

    plt.xlabel("Corruption Severity", fontweight="bold")
    plt.ylabel("Mean Corruption Accuracy (%)", fontweight="bold")
    plt.title(
        f"{MODEL_NAMES[model]}: Training Strategy Comparison",
        fontweight="bold",
    )
    plt.xticks([1, 2, 3, 4, 5])
    plt.legend(fontsize=int(24 * SCALE), ncol=2, frameon=True)
    plt.tight_layout()

    save_figure(PLOT_DIR / f"{model}_standard_vs_augmix_large")


# ============================================================
# Plot 4: TENT gain over frozen
# ============================================================

def plot_tent_gain(mean_df):
    plt.figure(figsize=(34, 20))

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

                merged["gain"] = (
                    merged["tent_accuracy"] - merged["frozen_accuracy"]
                ) * 100.0

                label = (
                    f"{MODEL_NAMES[model]} | "
                    f"{TRAINING_NAMES[training_type]} | "
                    f"{METHOD_NAMES[method]}"
                )

                plt.plot(
                    merged["severity"],
                    merged["gain"],
                    marker="o",
                    linewidth=7,
                    markersize=16,
                    label=label,
                )

    plt.axhline(0.0, linestyle="--", linewidth=5)

    plt.xlabel("Corruption Severity", fontweight="bold")
    plt.ylabel("Accuracy Gain over Frozen (percentage points)", fontweight="bold")
    plt.title("TENT Gain over Frozen Baseline", fontweight="bold")
    plt.xticks([1, 2, 3, 4, 5])
    plt.grid(True, alpha=0.45, linewidth=2.0)
    plt.legend(fontsize=int(24 * SCALE), ncol=2, frameon=True)
    plt.tight_layout()

    save_figure(PLOT_DIR / "tent_gain_over_frozen_all_large")


# ============================================================
# Plot 5: Continual TENT trajectory
# ============================================================

def plot_continual_trajectory(corr_df, severity):
    plt.figure(figsize=(42, 22))

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

            label = f"{MODEL_NAMES[model]} | {TRAINING_NAMES[training_type]}"

            plt.plot(
                range(len(CORRUPTIONS)),
                accuracy_to_percent(subset["accuracy"]),
                marker="o",
                linewidth=8,
                markersize=18,
                label=label,
            )

    ax = plt.gca()
    format_accuracy_axis(ax)

    plt.xlabel("Corruption Order", fontweight="bold")
    plt.ylabel("Accuracy (%)", fontweight="bold")
    plt.title(
        f"Continual TENT Trajectory at Severity {severity}",
        fontweight="bold",
    )

    short_labels = [SHORT_CORRUPTION_NAMES[c] for c in CORRUPTIONS]

    plt.xticks(
        range(len(CORRUPTIONS)),
        short_labels,
        rotation=35,
        ha="right",
        fontsize=int(30 * SCALE),
    )

    plt.legend(
        fontsize=int(28 * SCALE),
        ncol=2,
        frameon=True,
        loc="best",
    )

    plt.tight_layout()

    save_figure(PLOT_DIR / f"continual_trajectory_severity_{severity}_large")


# ============================================================
# Plot 6: Heatmap
# ============================================================

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

            row.append(float("nan") if len(values) == 0 else values.iloc[0] * 100.0)

        rows.append(row)

    heatmap_df = pd.DataFrame(
        rows,
        index=[SHORT_CORRUPTION_NAMES[c] for c in CORRUPTIONS],
        columns=["Frozen", "Episodic TENT", "Continual TENT"],
    )

    plt.figure(figsize=(22, 26))
    plt.imshow(heatmap_df.values, aspect="auto", vmin=0.0, vmax=100.0)

    cbar = plt.colorbar(label="Accuracy (%)")
    cbar.ax.tick_params(labelsize=int(30 * SCALE))
    cbar.set_label("Accuracy (%)", fontsize=int(34 * SCALE), fontweight="bold")

    plt.xticks(
        range(len(heatmap_df.columns)),
        heatmap_df.columns,
        rotation=25,
        ha="right",
        fontsize=int(32 * SCALE),
    )

    plt.yticks(
        range(len(heatmap_df.index)),
        heatmap_df.index,
        fontsize=int(30 * SCALE),
    )

    plt.title(
        f"{MODEL_NAMES[model]} | {TRAINING_NAMES[training_type]} | Severity {severity}",
        fontsize=int(38 * SCALE),
        fontweight="bold",
        pad=20,
    )

    for i in range(len(heatmap_df.index)):
        for j in range(len(methods)):
            value = heatmap_df.iloc[i, j]

            if pd.notna(value):
                plt.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=int(27 * SCALE),
                    fontweight="bold",
                )

    plt.tight_layout()

    save_figure(
        PLOT_DIR / f"{model}_{training_type}_heatmap_severity_{severity}_large"
    )


# ============================================================
# Plot 7: Entropy vs accuracy
# ============================================================

def plot_entropy_vs_accuracy(corr_df):
    subset = corr_df[
        corr_df["method"].isin(["episodic_tent", "continual_tent"])
        & corr_df["entropy"].notna()
    ]

    plt.figure(figsize=(34, 22))

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            model_subset = subset[
                (subset["model"] == model)
                & (subset["training_type"] == training_type)
            ]

            if len(model_subset) == 0:
                continue

            label = f"{MODEL_NAMES[model]} | {TRAINING_NAMES[training_type]}"

            plt.scatter(
                model_subset["entropy"],
                accuracy_to_percent(model_subset["accuracy"]),
                s=360,
                alpha=0.72,
                label=label,
                edgecolors="black",
                linewidths=1.2,
            )

    ax = plt.gca()
    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.45, linewidth=2.0)
    ax.tick_params(axis="both", labelsize=int(30 * SCALE))

    plt.xlabel("Prediction Entropy", fontweight="bold")
    plt.ylabel("Accuracy (%)", fontweight="bold")
    plt.title("Entropy vs Accuracy under TENT", fontweight="bold")

    plt.legend(
        fontsize=int(27 * SCALE),
        ncol=2,
        frameon=True,
        loc="best",
    )

    plt.tight_layout()

    save_figure(PLOT_DIR / "entropy_vs_accuracy_all_large")


# ============================================================
# Summary printing
# ============================================================

def print_summary(mean_df):
    summary_df = mean_df.copy()
    summary_df["mean_accuracy_percent"] = summary_df["mean_accuracy"] * 100.0

    print("\nMean Results")
    print(
        summary_df
        .sort_values(["model", "training_type", "method", "severity"])
        [["model", "training_type", "method", "severity", "mean_accuracy_percent"]]
        .to_string(index=False)
    )


# ============================================================
# Main
# ============================================================

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
    plot_severity_vs_accuracy_grid(mean_df)

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
