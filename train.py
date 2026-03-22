from __future__ import annotations

import argparse
import json
import shutil
from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml

from models.model_builder import build_classifier, set_backbone_trainable_layers
from utils.data_utils import create_datasets, save_class_names
from utils.visualize import (
    plot_confusion_matrix,
    plot_training_history,
    save_classification_report,
    save_history,
)


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def ensure_output_dirs(config: dict[str, Any]) -> tuple[Path, Path, Path]:
    paths_cfg = config["paths"]
    artifact_dir = Path(paths_cfg["artifact_dir"])
    log_dir = Path(paths_cfg["log_dir"])
    report_dir = Path(paths_cfg["report_dir"])

    artifact_dir.mkdir(parents=True, exist_ok=True)
    log_dir.mkdir(parents=True, exist_ok=True)
    report_dir.mkdir(parents=True, exist_ok=True)
    return artifact_dir, log_dir, report_dir


def compile_model(model: tf.keras.Model, learning_rate: float) -> None:
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )


def build_callbacks(
    checkpoint_path: Path,
    log_path: Path,
    config: dict[str, Any],
) -> list[tf.keras.callbacks.Callback]:
    training_cfg = config["training"]
    return [
        tf.keras.callbacks.ModelCheckpoint(
            filepath=str(checkpoint_path),
            monitor="val_accuracy",
            mode="max",
            save_best_only=True,
            verbose=1,
        ),
        tf.keras.callbacks.EarlyStopping(
            monitor="val_accuracy",
            mode="max",
            patience=int(training_cfg["early_stopping_patience"]),
            restore_best_weights=True,
            verbose=1,
        ),
        tf.keras.callbacks.ReduceLROnPlateau(
            monitor="val_loss",
            factor=float(training_cfg["reduce_lr_factor"]),
            patience=int(training_cfg["reduce_lr_patience"]),
            min_lr=1e-6,
            verbose=1,
        ),
        tf.keras.callbacks.CSVLogger(str(log_path)),
    ]


def merge_histories(*histories: dict[str, list[float]]) -> dict[str, list[float]]:
    merged: dict[str, list[float]] = {}
    for history in histories:
        for key, values in history.items():
            merged.setdefault(key, []).extend(float(value) for value in values)
    return merged


def evaluate_model(model: tf.keras.Model, dataset: tf.data.Dataset) -> tuple[float, float]:
    loss, accuracy = model.evaluate(dataset, verbose=0)
    return float(loss), float(accuracy)


def select_best_model(
    artifact_dir: Path,
    validation_dataset: tf.data.Dataset,
) -> tuple[Path, dict[str, float]]:
    candidates = {
        "stage1": artifact_dir / "stage1_best.keras",
        "stage2": artifact_dir / "stage2_best.keras",
    }
    best_stage = ""
    best_path: Path | None = None
    best_metrics = {"val_loss": float("inf"), "val_accuracy": float("-inf")}

    for stage_name, model_path in candidates.items():
        if not model_path.exists():
            continue
        model = tf.keras.models.load_model(model_path)
        val_loss, val_accuracy = evaluate_model(model, validation_dataset)
        if val_accuracy > best_metrics["val_accuracy"]:
            best_stage = stage_name
            best_path = model_path
            best_metrics = {"val_loss": val_loss, "val_accuracy": val_accuracy}

    if best_path is None:
        raise FileNotFoundError("未找到可用的阶段模型检查点。")

    final_model_path = artifact_dir / "best_model.keras"
    if best_path.resolve() != final_model_path.resolve():
        shutil.copy2(best_path, final_model_path)

    metrics_path = artifact_dir / "best_model_metrics.json"
    metrics_path.write_text(
        json.dumps(
            {
                "selected_stage": best_stage,
                **best_metrics,
            },
            ensure_ascii=False,
            indent=2,
        ),
        encoding="utf-8",
    )
    return final_model_path, best_metrics


def collect_predictions(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[list[int], list[int]]:
    y_true: list[int] = []
    y_pred: list[int] = []

    for images, labels in dataset:
        probabilities = model.predict(images, verbose=0)
        y_true.extend(int(label) for label in labels.numpy())
        y_pred.extend(int(index) for index in tf.argmax(probabilities, axis=1).numpy())

    return y_true, y_pred


def main(config_path: Path) -> None:
    config = load_config(config_path)
    seed = int(config["data"].get("seed", 42))
    tf.keras.utils.set_random_seed(seed)

    artifact_dir, log_dir, report_dir = ensure_output_dirs(config)
    train_ds, val_ds, class_names, class_weights = create_datasets(config)
    save_class_names(class_names, artifact_dir / "class_names.json")

    model, base_model = build_classifier(config)
    compile_model(model, float(config["training"]["stage1_learning_rate"]))

    print("开始阶段 1 训练：冻结 backbone。")
    stage1_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(config["training"]["stage1_epochs"]),
        class_weight=class_weights,
        callbacks=build_callbacks(
            artifact_dir / "stage1_best.keras",
            log_dir / "stage1_training.csv",
            config,
        ),
        verbose=1,
    )

    print("开始阶段 2 训练：解冻最后若干层进行微调。")
    set_backbone_trainable_layers(base_model, int(config["training"]["fine_tune_layers"]))
    compile_model(model, float(config["training"]["stage2_learning_rate"]))

    stage2_history = model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=int(config["training"]["stage2_epochs"]),
        class_weight=class_weights,
        callbacks=build_callbacks(
            artifact_dir / "stage2_best.keras",
            log_dir / "stage2_training.csv",
            config,
        ),
        verbose=1,
    )

    combined_history = merge_histories(stage1_history.history, stage2_history.history)
    save_history(combined_history, report_dir / "history.json")
    plot_training_history(combined_history, report_dir / "training_curves.png")

    best_model_path, best_metrics = select_best_model(artifact_dir, val_ds)
    best_model = tf.keras.models.load_model(best_model_path)
    y_true, y_pred = collect_predictions(best_model, val_ds)

    plot_confusion_matrix(y_true, y_pred, class_names, report_dir / "confusion_matrix.png")
    save_classification_report(
        y_true,
        y_pred,
        class_names,
        report_dir / "classification_report.txt",
    )

    print(f"训练完成，最佳模型已保存到: {best_model_path}")
    print(
        f"验证集指标: loss={best_metrics['val_loss']:.4f}, "
        f"accuracy={best_metrics['val_accuracy']:.4f}"
    )


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="训练 TensorFlow 2.x 轻量级三分类模型")
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="配置文件路径",
    )
    args = parser.parse_args()
    main(args.config)
