from __future__ import annotations

import json
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from sklearn.metrics import classification_report, confusion_matrix


def plot_training_history(history: dict[str, list[float]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    axes[0].plot(history.get("accuracy", []), label="train_accuracy")
    axes[0].plot(history.get("val_accuracy", []), label="val_accuracy")
    axes[0].set_title("Accuracy")
    axes[0].set_xlabel("Epoch")
    axes[0].set_ylabel("Accuracy")
    axes[0].legend()

    axes[1].plot(history.get("loss", []), label="train_loss")
    axes[1].plot(history.get("val_loss", []), label="val_loss")
    axes[1].set_title("Loss")
    axes[1].set_xlabel("Epoch")
    axes[1].set_ylabel("Loss")
    axes[1].legend()

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confusion_matrix(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)

    fig, ax = plt.subplots(figsize=(6, 5))
    image = ax.imshow(matrix, cmap="Blues")
    ax.figure.colorbar(image, ax=ax)

    ax.set_xticks(np.arange(len(class_names)))
    ax.set_yticks(np.arange(len(class_names)))
    ax.set_xticklabels(class_names)
    ax.set_yticklabels(class_names)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title("Confusion Matrix")

    threshold = matrix.max() / 2.0 if matrix.size else 0.0
    for i in range(matrix.shape[0]):
        for j in range(matrix.shape[1]):
            ax.text(
                j,
                i,
                format(matrix[i, j], "d"),
                ha="center",
                va="center",
                color="white" if matrix[i, j] > threshold else "black",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_classification_report(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    report = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        zero_division=0,
    )
    save_path.write_text(report, encoding="utf-8")


def save_history(history: dict[str, list[float]], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(
        json.dumps(history, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_per_class_accuracy(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    matrix = confusion_matrix(y_true, y_pred)
    denominators = np.maximum(matrix.sum(axis=1), 1)
    accuracies = matrix.diagonal() / denominators

    fig, ax = plt.subplots(figsize=(7, 4))
    bars = ax.bar(class_names, accuracies, color="#4c72b0")
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Accuracy")
    ax.set_title("Per-Class Accuracy")

    for bar, accuracy in zip(bars, accuracies):
        ax.text(
            bar.get_x() + bar.get_width() / 2.0,
            accuracy + 0.02,
            f"{accuracy:.3f}",
            ha="center",
            va="bottom",
        )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_confidence_histogram(confidences: list[float], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(7, 4))
    ax.hist(confidences, bins=20, range=(0.0, 1.0), color="#dd8452", edgecolor="white")
    ax.set_xlabel("Top-1 Confidence")
    ax.set_ylabel("Count")
    ax.set_title("Confidence Distribution")
    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def save_metrics(metrics: dict[str, object], save_path: Path) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    save_path.write_text(
        json.dumps(metrics, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def plot_metric_comparison(
    keras_metrics: dict[str, object],
    tflite_metrics: dict[str, object],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    metric_labels = ["Accuracy", "Macro F1", "Weighted F1"]
    keras_values = [
        float(keras_metrics["accuracy"]),
        float(keras_metrics["macro_avg"]["f1_score"]),
        float(keras_metrics["weighted_avg"]["f1_score"]),
    ]
    tflite_values = [
        float(tflite_metrics["accuracy"]),
        float(tflite_metrics["macro_avg"]["f1_score"]),
        float(tflite_metrics["weighted_avg"]["f1_score"]),
    ]

    positions = np.arange(len(metric_labels))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    keras_bars = ax.bar(positions - width / 2, keras_values, width=width, label="Keras")
    tflite_bars = ax.bar(positions + width / 2, tflite_values, width=width, label="TFLite")

    ax.set_xticks(positions)
    ax.set_xticklabels(metric_labels)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Score")
    ax.set_title("Keras vs TFLite Metrics")
    ax.legend()

    for bars in [keras_bars, tflite_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)


def plot_per_class_accuracy_comparison(
    y_true: list[int],
    keras_pred: list[int],
    tflite_pred: list[int],
    class_names: list[str],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)

    keras_matrix = confusion_matrix(y_true, keras_pred)
    tflite_matrix = confusion_matrix(y_true, tflite_pred)
    keras_denominators = np.maximum(keras_matrix.sum(axis=1), 1)
    tflite_denominators = np.maximum(tflite_matrix.sum(axis=1), 1)
    keras_accuracy = keras_matrix.diagonal() / keras_denominators
    tflite_accuracy = tflite_matrix.diagonal() / tflite_denominators

    positions = np.arange(len(class_names))
    width = 0.35

    fig, ax = plt.subplots(figsize=(8, 4.5))
    keras_bars = ax.bar(positions - width / 2, keras_accuracy, width=width, label="Keras")
    tflite_bars = ax.bar(positions + width / 2, tflite_accuracy, width=width, label="TFLite")

    ax.set_xticks(positions)
    ax.set_xticklabels(class_names)
    ax.set_ylim(0.0, 1.0)
    ax.set_ylabel("Per-Class Accuracy")
    ax.set_title("Per-Class Accuracy Comparison")
    ax.legend()

    for bars in [keras_bars, tflite_bars]:
        for bar in bars:
            height = bar.get_height()
            ax.text(
                bar.get_x() + bar.get_width() / 2.0,
                height + 0.02,
                f"{height:.3f}",
                ha="center",
                va="bottom",
            )

    fig.tight_layout()
    fig.savefig(save_path, dpi=200)
    plt.close(fig)
