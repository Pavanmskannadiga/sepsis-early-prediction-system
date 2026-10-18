"""
SHAP Visualization Module.

Generates publication-quality SHAP plots for the sepsis prediction model:
  - Feature importance bar chart
  - SHAP summary (beeswarm) plot
  - Temporal SHAP heatmap
  - Patient-level waterfall plots
  - SHAP dependence plots for top features
"""

import sys
import numpy as np
import matplotlib
matplotlib.use("Agg")  # Non-interactive backend for saving
import matplotlib.pyplot as plt
import matplotlib.colors as mcolors
from pathlib import Path
from typing import List, Tuple, Optional, Dict

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
import config


# Consistent plot styling
plt.rcParams.update({
    "figure.facecolor": "#0f0f1a",
    "axes.facecolor": "#1a1a2e",
    "text.color": "#e0e0e0",
    "axes.labelcolor": "#e0e0e0",
    "xtick.color": "#b0b0b0",
    "ytick.color": "#b0b0b0",
    "axes.edgecolor": "#333366",
    "grid.color": "#252550",
    "grid.alpha": 0.3,
    "font.family": "sans-serif",
    "font.size": 11,
})

# Color palette
COLORS = {
    "primary": "#667eea",
    "secondary": "#764ba2",
    "accent": "#00d2ff",
    "positive": "#ef4444",
    "negative": "#3b82f6",
    "gradient_start": "#667eea",
    "gradient_end": "#ef4444",
}


def plot_feature_importance(
    feature_importance: List[Tuple[str, float]],
    title: str = "Top Features by SHAP Importance",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (10, 8)
) -> plt.Figure:
    """
    Create a horizontal bar chart of feature importance.

    Args:
        feature_importance: List of (feature_name, importance_score) tuples.
        title: Plot title.
        save_path: Path to save the figure.
        figsize: Figure size.

    Returns:
        Matplotlib Figure object.
    """
    # Filter out mask features for cleaner visualization
    filtered = [(n, s) for n, s in feature_importance if "_mask" not in n]
    if not filtered:
        filtered = feature_importance

    names = [n for n, s in filtered]
    scores = [s for n, s in filtered]

    fig, ax = plt.subplots(figsize=figsize)

    # Create gradient colors
    n = len(names)
    colors = [plt.cm.RdYlBu_r(i / max(n - 1, 1)) for i in range(n)]

    # Horizontal bar chart (reversed for top-to-bottom order)
    y_pos = np.arange(len(names))
    bars = ax.barh(y_pos, scores[::-1], color=colors[::-1], edgecolor="none", height=0.7)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names[::-1], fontsize=12)
    ax.set_xlabel("Mean |SHAP Value|", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.invert_yaxis()  # Top feature at top

    # Add value labels
    for bar, score in zip(bars, scores[::-1]):
        ax.text(bar.get_width() + max(scores) * 0.02, bar.get_y() + bar.get_height() / 2,
                f"{score:.4f}", va="center", fontsize=10, color="#e0e0e0")

    ax.grid(axis="x", alpha=0.2)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.SHAPConfig.dpi,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Feature importance plot saved to: {save_path}")

    return fig


def plot_temporal_heatmap(
    temporal_importance: np.ndarray,
    feature_names: List[str],
    top_k: int = 15,
    title: str = "Feature Importance Over Time (Temporal SHAP)",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 8)
) -> plt.Figure:
    """
    Create a heatmap showing how feature importance changes over the
    observation window. X-axis = hours before prediction, Y-axis = features.

    Args:
        temporal_importance: Shape (seq_len, n_features).
        feature_names: Feature names.
        top_k: Show only top-k most important features.
        title: Plot title.
        save_path: Path to save.
        figsize: Figure size.

    Returns:
        Matplotlib Figure.
    """
    # Filter out mask features
    clinical_indices = [i for i, name in enumerate(feature_names) if "_mask" not in name]
    clinical_names = [feature_names[i] for i in clinical_indices]
    data = temporal_importance[:, clinical_indices]

    # Select top-k features by overall importance
    overall_importance = data.mean(axis=0)
    top_indices = np.argsort(overall_importance)[::-1][:top_k]
    data = data[:, top_indices]
    names = [clinical_names[i] for i in top_indices]

    fig, ax = plt.subplots(figsize=figsize)

    # Custom colormap: dark blue → cyan → yellow → red
    cmap = mcolors.LinearSegmentedColormap.from_list(
        "shap_temporal", ["#0a0a2e", "#1a237e", "#00bcd4", "#ffeb3b", "#ff5722"]
    )

    seq_len = data.shape[0]
    im = ax.imshow(
        data.T, aspect="auto", cmap=cmap,
        extent=[0, seq_len, len(names) - 0.5, -0.5]
    )

    ax.set_yticks(range(len(names)))
    ax.set_yticklabels(names, fontsize=11)
    ax.set_xlabel(f"Hours in Observation Window (0 = earliest, {seq_len} = prediction time)", fontsize=12)
    ax.set_title(title, fontsize=14, fontweight="bold", pad=15)

    # Colorbar
    cbar = plt.colorbar(im, ax=ax, fraction=0.02, pad=0.04)
    cbar.set_label("Mean |SHAP Value|", fontsize=11)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.SHAPConfig.dpi,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Temporal heatmap saved to: {save_path}")

    return fig


def plot_roc_comparison(
    roc_data: Dict[str, Dict],
    title: str = "ROC Curve Comparison",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (8, 8)
) -> plt.Figure:
    """
    Plot ROC curves for multiple models on the same axes.

    Args:
        roc_data: Dict mapping model name to dict with 'fpr', 'tpr', 'auc'.
        title: Plot title.
        save_path: Save path.
        figsize: Figure size.

    Returns:
        Figure.
    """
    fig, ax = plt.subplots(figsize=figsize)

    model_colors = ["#667eea", "#ef4444", "#10b981", "#f59e0b", "#8b5cf6", "#ec4899"]

    for i, (name, data) in enumerate(roc_data.items()):
        color = model_colors[i % len(model_colors)]
        ax.plot(
            data["fpr"], data["tpr"],
            color=color, linewidth=2.5,
            label=f'{name} (AUC = {data["auc"]:.3f})'
        )

    # Diagonal reference line
    ax.plot([0, 1], [0, 1], "w--", alpha=0.3, linewidth=1)

    ax.set_xlim([0, 1])
    ax.set_ylim([0, 1.02])
    ax.set_xlabel("False Positive Rate", fontsize=13)
    ax.set_ylabel("True Positive Rate", fontsize=13)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)
    ax.legend(loc="lower right", fontsize=11, framealpha=0.3)
    ax.grid(True, alpha=0.15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.SHAPConfig.dpi,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"ROC comparison saved to: {save_path}")

    return fig


def plot_training_history(
    history: Dict[str, list],
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (14, 5)
) -> plt.Figure:
    """
    Plot training history: loss curves and validation metrics.

    Args:
        history: Training history dictionary from SepsisTrainer.
        save_path: Save path.
        figsize: Figure size.

    Returns:
        Figure.
    """
    fig, axes = plt.subplots(1, 3, figsize=figsize)

    epochs = range(1, len(history["train_loss"]) + 1)

    # Loss curves
    axes[0].plot(epochs, history["train_loss"], color="#667eea", linewidth=2, label="Train")
    axes[0].plot(epochs, history["val_loss"], color="#ef4444", linewidth=2, label="Validation")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Loss")
    axes[0].set_title("Training & Validation Loss")
    axes[0].legend(framealpha=0.3)
    axes[0].grid(True, alpha=0.15)

    # AUROC
    axes[1].plot(epochs, history["val_auroc"], color="#10b981", linewidth=2)
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("AUROC")
    axes[1].set_title("Validation AUROC")
    axes[1].grid(True, alpha=0.15)

    # Learning rate
    axes[2].plot(epochs, history["learning_rate"], color="#f59e0b", linewidth=2)
    axes[2].set_xlabel("Epoch")
    axes[2].set_ylabel("Learning Rate")
    axes[2].set_title("Learning Rate Schedule")
    axes[2].set_yscale("log")
    axes[2].grid(True, alpha=0.15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.SHAPConfig.dpi,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Training history saved to: {save_path}")

    return fig


def plot_confusion_matrix(
    metrics: Dict[str, float],
    title: str = "Confusion Matrix",
    save_path: Optional[str] = None,
    figsize: Tuple[int, int] = (7, 6)
) -> plt.Figure:
    """
    Plot a styled confusion matrix from metrics dictionary.

    Args:
        metrics: Dict with 'tp', 'fp', 'fn', 'tn' keys.
        title: Plot title.
        save_path: Save path.
        figsize: Figure size.

    Returns:
        Figure.
    """
    cm = np.array([
        [metrics["tn"], metrics["fp"]],
        [metrics["fn"], metrics["tp"]]
    ])

    fig, ax = plt.subplots(figsize=figsize)

    cmap = mcolors.LinearSegmentedColormap.from_list(
        "cm_cmap", ["#1a1a2e", "#667eea", "#ef4444"]
    )

    im = ax.imshow(cm, cmap=cmap, aspect="auto")

    # Add text annotations
    for i in range(2):
        for j in range(2):
            val = cm[i, j]
            color = "white" if val > cm.max() * 0.5 else "#e0e0e0"
            label = ["TN", "FP", "FN", "TP"][i * 2 + j]
            ax.text(j, i, f"{label}\n{val:,}", ha="center", va="center",
                    fontsize=16, fontweight="bold", color=color)

    ax.set_xticks([0, 1])
    ax.set_yticks([0, 1])
    ax.set_xticklabels(["Predicted\nNo Sepsis", "Predicted\nSepsis"], fontsize=12)
    ax.set_yticklabels(["Actual\nNo Sepsis", "Actual\nSepsis"], fontsize=12)
    ax.set_title(title, fontsize=15, fontweight="bold", pad=15)

    plt.tight_layout()

    if save_path:
        fig.savefig(save_path, dpi=config.SHAPConfig.dpi,
                    bbox_inches="tight", facecolor=fig.get_facecolor())
        print(f"Confusion matrix saved to: {save_path}")

    return fig
