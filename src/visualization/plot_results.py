import json
import os
from pathlib import Path

import matplotlib.pyplot as plt
import pandas as pd


# ============================================================
# TWO-COLUMN PAPER-READABLE GLOBAL PLOT STYLE
# ============================================================
# Bu ayarlar özellikle figürler LaTeX/Word'de iki kolona veya
# tam sayfa genişliğine sıkıştırıldığında okunabilir kalması için büyütüldü.
#
# Eğer figürler hâlâ küçük görünürse:
#   SCALE = 1.15 yap
#
# Eğer figürler çok büyük görünürse:
#   SCALE = 0.90 yap
# ============================================================

SCALE = 1.00

FONT_BASE = int(54 * SCALE)
FONT_TITLE = int(68 * SCALE)
FONT_LABEL = int(62 * SCALE)
FONT_TICK = int(48 * SCALE)
FONT_LEGEND = int(44 * SCALE)
FONT_SUPER = int(68 * SCALE)
FONT_ANNOT = int(42 * SCALE)

LINE_WIDTH = 9
MARKER_SIZE = 22
AXIS_WIDTH = 3.4
GRID_WIDTH = 2.4
TICK_WIDTH = 3.0
TICK_SIZE = 12

plt.rcParams.update({
    "font.family": "DejaVu Sans",

    "font.size": FONT_BASE,
    "axes.titlesize": FONT_TITLE,
    "axes.labelsize": FONT_LABEL,
    "xtick.labelsize": FONT_TICK,
    "ytick.labelsize": FONT_TICK,
    "legend.fontsize": FONT_LEGEND,
    "figure.titlesize": FONT_SUPER,

    "axes.titleweight": "bold",
    "axes.labelweight": "bold",

    "lines.linewidth": LINE_WIDTH,
    "lines.markersize": MARKER_SIZE,

    "axes.linewidth": AXIS_WIDTH,
    "xtick.major.width": TICK_WIDTH,
    "ytick.major.width": TICK_WIDTH,
    "xtick.major.size": TICK_SIZE,
    "ytick.major.size": TICK_SIZE,

    "savefig.dpi": 600,
    "figure.dpi": 160,

    # PDF çıktıda fontların düzgün gömülmesi için.
    # Makaleye koyunca fontlar daha temiz kalır.
    "pdf.fonttype": 42,
    "ps.fonttype": 42,

    # Eksi işaretinin bozulmasını engeller.
    "axes.unicode_minus": False,
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

# Eski klasörü ezmemek için yeni klasör.
PLOT_DIR = RESULT_ROOT / "plots_two_column_readable"

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

    PDF is recommended for papers because it stays sharp when enlarged
    or compressed in two-column layouts.
    """
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)

    png_path = path.with_suffix(".png")
    pdf_path = path.with_suffix(".pdf")

    plt.savefig(
        png_path,
        dpi=600,
        bbox_inches="tight",
        pad_inches=0.28,
        facecolor="white",
    )
    plt.savefig(
        pdf_path,
        bbox_inches="tight",
        pad_inches=0.28,
        facecolor="white",
    )
    plt.close()

    print(f"Saved: {png_path}")
    print(f"Saved: {pdf_path}")


def make_tick_labels_bold(ax):
    for label in ax.get_xticklabels():
        label.set_fontweight("bold")
    for label in ax.get_yticklabels():
        label.set_fontweight("bold")


def thicken_spines(ax):
    for spine in ax.spines.values():
        spine.set_linewidth(AXIS_WIDTH)


def format_accuracy_axis(ax, ymin=0, ymax=100):
    """
    Use percentage scale instead of 0-1 scale.
    """
    ax.set_ylim(ymin, ymax)
    ax.set_yticks([0, 20, 40, 60, 80, 100])
    ax.grid(True, alpha=0.45, linewidth=GRID_WIDTH)
    ax.tick_params(
        axis="both",
        labelsize=FONT_TICK,
        width=TICK_WIDTH,
        length=TICK_SIZE,
    )
    make_tick_labels_bold(ax)
    thicken_spines(ax)


def format_general_axis(ax):
    ax.grid(True, alpha=0.45, linewidth=GRID_WIDTH)
    ax.tick_params(
        axis="both",
        labelsize=FONT_TICK,
        width=TICK_WIDTH,
        length=TICK_SIZE,
    )
    make_tick_labels_bold(ax)
    thicken_spines(ax)


def accuracy_to_percent(series):
    return series * 100.0


def add_big_legend_below(fig, ax, ncol=2, y=-0.02):
    """
    Puts legend below the figure so that the plot area remains readable.
    Especially useful when the final figure is squeezed into two columns.
    """
    handles, labels = ax.get_legend_handles_labels()

    if len(handles) == 0:
        return

    legend = fig.legend(
        handles,
        labels,
        loc="lower center",
        bbox_to_anchor=(0.5, y),
        ncol=ncol,
        frameon=True,
        fontsize=FONT_LEGEND,
        handlelength=2.8,
        columnspacing=1.2,
        handletextpad=0.6,
        borderpad=0.7,
    )

    legend.get_frame().set_linewidth(2.5)
    legend.get_frame().set_alpha(0.95)


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
    fig, ax = plt.subplots(figsize=(40, 24))

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

                ax.plot(
                    subset["severity"],
                    accuracy_to_percent(subset["mean_accuracy"]),
                    marker="o",
                    linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE,
                    label=label,
                )

    format_accuracy_axis(ax)

    ax.set_xlabel("Corruption Severity", fontweight="bold", labelpad=20)
    ax.set_ylabel("Mean Corruption Accuracy (%)", fontweight="bold", labelpad=20)
    ax.set_title(
        "Mean Accuracy across Corruption Severity",
        fontweight="bold",
        pad=28,
    )
    ax.set_xticks([1, 2, 3, 4, 5])

    add_big_legend_below(fig, ax, ncol=3, y=-0.08)

    fig.tight_layout(rect=[0, 0.15, 1, 1])

    save_figure(PLOT_DIR / "severity_vs_accuracy_all_two_column")


# ============================================================
# Plot 2: Grid plot by model and training type
# ============================================================

def plot_severity_vs_accuracy_grid(mean_df):
    fig, axes = plt.subplots(
        nrows=2,
        ncols=3,
        figsize=(46, 28),
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
                    linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE,
                    label=style["label"],
                )

            ax.set_title(
                TRAINING_NAMES[training_type],
                fontsize=FONT_TITLE,
                fontweight="bold",
                pad=22,
            )

            ax.set_xticks([1, 2, 3, 4, 5])
            format_accuracy_axis(ax)

            if col == 0:
                ax.set_ylabel(
                    f"{MODEL_NAMES[model]}\nAccuracy (%)",
                    fontsize=FONT_LABEL,
                    fontweight="bold",
                    labelpad=22,
                )

    handles, labels = axes[0, 0].get_legend_handles_labels()

    legend = fig.legend(
        handles,
        labels,
        loc="upper center",
        ncol=3,
        frameon=True,
        fontsize=FONT_LEGEND,
        bbox_to_anchor=(0.5, 1.03),
        handlelength=3.0,
        columnspacing=1.4,
        handletextpad=0.7,
        borderpad=0.7,
    )
    legend.get_frame().set_linewidth(2.5)

    fig.supxlabel(
        "Corruption Severity",
        fontsize=FONT_SUPER,
        fontweight="bold",
        y=0.035,
    )

    fig.supylabel(
        "Mean CIFAR-10-C Accuracy (%)",
        fontsize=FONT_SUPER,
        fontweight="bold",
        x=0.005,
    )

    fig.tight_layout(rect=[0.045, 0.075, 1, 0.92])

    save_figure(PLOT_DIR / "severity_vs_accuracy_grid_two_column")


# ============================================================
# Plot 3: Model-specific comparison
# ============================================================

def plot_model_training_comparison(mean_df, model):
    fig, ax = plt.subplots(figsize=(38, 24))

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

            ax.plot(
                subset["severity"],
                accuracy_to_percent(subset["mean_accuracy"]),
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                label=label,
            )

    format_accuracy_axis(ax)

    ax.set_xlabel("Corruption Severity", fontweight="bold", labelpad=20)
    ax.set_ylabel("Mean Corruption Accuracy (%)", fontweight="bold", labelpad=20)
    ax.set_title(
        f"{MODEL_NAMES[model]}: Training Strategy Comparison",
        fontweight="bold",
        pad=28,
    )
    ax.set_xticks([1, 2, 3, 4, 5])

    add_big_legend_below(fig, ax, ncol=3, y=-0.055)

    fig.tight_layout(rect=[0, 0.13, 1, 1])

    save_figure(PLOT_DIR / f"{model}_standard_vs_augmix_two_column")


# ============================================================
# Plot 4: TENT gain over frozen
# ============================================================

def plot_tent_gain(mean_df):
    fig, ax = plt.subplots(figsize=(40, 24))

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

                ax.plot(
                    merged["severity"],
                    merged["gain"],
                    marker="o",
                    linewidth=LINE_WIDTH,
                    markersize=MARKER_SIZE,
                    label=label,
                )

    ax.axhline(0.0, linestyle="--", linewidth=6)

    ax.set_xlabel("Corruption Severity", fontweight="bold", labelpad=20)
    ax.set_ylabel(
        "Accuracy Gain over Frozen (percentage points)",
        fontweight="bold",
        labelpad=20,
    )
    ax.set_title("TENT Gain over Frozen Baseline", fontweight="bold", pad=28)
    ax.set_xticks([1, 2, 3, 4, 5])

    format_general_axis(ax)

    add_big_legend_below(fig, ax, ncol=3, y=-0.07)

    fig.tight_layout(rect=[0, 0.15, 1, 1])

    save_figure(PLOT_DIR / "tent_gain_over_frozen_all_two_column")


# ============================================================
# Plot 5: Continual TENT trajectory
# ============================================================

def plot_continual_trajectory(corr_df, severity):
    fig, ax = plt.subplots(figsize=(46, 26))

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

            # reindex kullanımı .loc[CORRUPTIONS] gibi eksik corruption varsa
            # hata vermesini engeller.
            subset = (
                subset
                .set_index("corruption")
                .reindex(CORRUPTIONS)
                .reset_index()
            )

            subset = subset.dropna(subset=["accuracy"])

            if len(subset) == 0:
                continue

            label = f"{MODEL_NAMES[model]} | {TRAINING_NAMES[training_type]}"

            x_positions = [
                CORRUPTIONS.index(corruption)
                for corruption in subset["corruption"]
            ]

            ax.plot(
                x_positions,
                accuracy_to_percent(subset["accuracy"]),
                marker="o",
                linewidth=LINE_WIDTH,
                markersize=MARKER_SIZE,
                label=label,
            )

    format_accuracy_axis(ax)

    ax.set_xlabel("Corruption Order", fontweight="bold", labelpad=20)
    ax.set_ylabel("Accuracy (%)", fontweight="bold", labelpad=20)
    ax.set_title(
        f"Continual TENT Trajectory at Severity {severity}",
        fontweight="bold",
        pad=28,
    )

    short_labels = [SHORT_CORRUPTION_NAMES[c] for c in CORRUPTIONS]

    ax.set_xticks(range(len(CORRUPTIONS)))
    ax.set_xticklabels(
        short_labels,
        rotation=42,
        ha="right",
        fontsize=FONT_TICK,
        fontweight="bold",
    )

    add_big_legend_below(fig, ax, ncol=2, y=-0.055)

    fig.tight_layout(rect=[0, 0.13, 1, 1])

    save_figure(PLOT_DIR / f"continual_trajectory_severity_{severity}_two_column")


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

    fig, ax = plt.subplots(figsize=(24, 30))

    image = ax.imshow(
        heatmap_df.values,
        aspect="auto",
        vmin=0.0,
        vmax=100.0,
    )

    cbar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    cbar.ax.tick_params(
        labelsize=FONT_TICK,
        width=TICK_WIDTH,
        length=TICK_SIZE,
    )
    cbar.set_label(
        "Accuracy (%)",
        fontsize=FONT_LABEL,
        fontweight="bold",
        labelpad=18,
    )

    for tick_label in cbar.ax.get_yticklabels():
        tick_label.set_fontweight("bold")

    ax.set_xticks(range(len(heatmap_df.columns)))
    ax.set_xticklabels(
        heatmap_df.columns,
        rotation=25,
        ha="right",
        fontsize=FONT_TICK,
        fontweight="bold",
    )

    ax.set_yticks(range(len(heatmap_df.index)))
    ax.set_yticklabels(
        heatmap_df.index,
        fontsize=FONT_TICK,
        fontweight="bold",
    )

    ax.set_title(
        f"{MODEL_NAMES[model]} | {TRAINING_NAMES[training_type]} | Severity {severity}",
        fontsize=FONT_TITLE,
        fontweight="bold",
        pad=28,
    )

    for i in range(len(heatmap_df.index)):
        for j in range(len(methods)):
            value = heatmap_df.iloc[i, j]

            if pd.notna(value):
                ax.text(
                    j,
                    i,
                    f"{value:.1f}",
                    ha="center",
                    va="center",
                    fontsize=FONT_ANNOT,
                    fontweight="bold",
                )

    ax.tick_params(
        axis="both",
        width=TICK_WIDTH,
        length=TICK_SIZE,
    )
    thicken_spines(ax)

    fig.tight_layout()

    save_figure(
        PLOT_DIR / f"{model}_{training_type}_heatmap_severity_{severity}_two_column"
    )


# ============================================================
# Plot 7: Entropy vs accuracy
# ============================================================

def plot_entropy_vs_accuracy(corr_df):
    subset = corr_df[
        corr_df["method"].isin(["episodic_tent", "continual_tent"])
        & corr_df["entropy"].notna()
    ]

    fig, ax = plt.subplots(figsize=(38, 26))

    for model in MODELS:
        for training_type in TRAINING_TYPES:
            model_subset = subset[
                (subset["model"] == model)
                & (subset["training_type"] == training_type)
            ]

            if len(model_subset) == 0:
                continue

            label = f"{MODEL_NAMES[model]} | {TRAINING_NAMES[training_type]}"

            ax.scatter(
                model_subset["entropy"],
                accuracy_to_percent(model_subset["accuracy"]),
                s=520,
                alpha=0.72,
                label=label,
                edgecolors="black",
                linewidths=2.2,
            )

    ax.set_ylim(0, 100)
    ax.set_yticks([0, 20, 40, 60, 80, 100])

    ax.set_xlabel("Prediction Entropy", fontweight="bold", labelpad=20)
    ax.set_ylabel("Accuracy (%)", fontweight="bold", labelpad=20)
    ax.set_title("Entropy vs Accuracy under TENT", fontweight="bold", pad=28)

    format_general_axis(ax)

    add_big_legend_below(fig, ax, ncol=2, y=-0.055)

    fig.tight_layout(rect=[0, 0.13, 1, 1])

    save_figure(PLOT_DIR / "entropy_vs_accuracy_all_two_column")


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
