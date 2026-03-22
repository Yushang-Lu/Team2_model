from __future__ import annotations

import argparse
import csv
from pathlib import Path

import tensorflow as tf

from utils.data_utils import create_evaluation_dataset, load_class_names
from utils.eval_utils import (
    build_metrics,
    build_prediction_comparison,
    load_config,
    resolve_class_names_path,
    run_keras_inference,
    run_tflite_inference,
)
from utils.visualize import (
    plot_confusion_matrix,
    plot_metric_comparison,
    plot_per_class_accuracy_comparison,
    save_classification_report,
    save_metrics,
)


def resolve_output_dir(config: dict[str, object], explicit_dir: Path | None) -> Path:
    if explicit_dir is not None:
        return explicit_dir
    paths_cfg = config.get("paths", {})
    return Path(paths_cfg.get("compare_report_dir", "reports/compare"))


def save_disagreement_csv(
    y_true: list[int],
    keras_pred: list[int],
    tflite_pred: list[int],
    keras_confidence: list[float],
    tflite_confidence: list[float],
    class_names: list[str],
    save_path: Path,
) -> None:
    save_path.parent.mkdir(parents=True, exist_ok=True)
    with save_path.open("w", encoding="utf-8", newline="") as file:
        writer = csv.writer(file)
        writer.writerow(
            [
                "sample_index",
                "true_label",
                "keras_prediction",
                "tflite_prediction",
                "keras_confidence",
                "tflite_confidence",
                "keras_correct",
                "tflite_correct",
            ]
        )

        for index, (true_index, keras_index, tflite_index, keras_score, tflite_score) in enumerate(
            zip(y_true, keras_pred, tflite_pred, keras_confidence, tflite_confidence)
        ):
            if keras_index == tflite_index:
                continue

            writer.writerow(
                [
                    index,
                    class_names[true_index],
                    class_names[keras_index],
                    class_names[tflite_index],
                    f"{keras_score:.6f}",
                    f"{tflite_score:.6f}",
                    int(keras_index == true_index),
                    int(tflite_index == true_index),
                ]
            )


def main(
    keras_model_path: Path,
    tflite_model_path: Path,
    config_path: Path,
    output_dir: Path | None,
    class_names_path: Path | None,
    num_threads: int,
) -> None:
    if not keras_model_path.exists():
        raise FileNotFoundError(f"Keras 模型文件不存在: {keras_model_path}")
    if not tflite_model_path.exists():
        raise FileNotFoundError(f"TFLite 模型文件不存在: {tflite_model_path}")

    config = load_config(config_path)
    class_names = load_class_names(resolve_class_names_path(keras_model_path, class_names_path))
    dataset, source_name = create_evaluation_dataset(config, class_names)
    report_dir = resolve_output_dir(config, output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    print("开始评估 Keras 模型。")
    keras_model = tf.keras.models.load_model(keras_model_path)
    keras_true, keras_pred, keras_confidence = run_keras_inference(keras_model, dataset)
    keras_metrics = build_metrics(keras_true, keras_pred, class_names, source_name)
    save_classification_report(
        keras_true,
        keras_pred,
        class_names,
        report_dir / "keras_classification_report.txt",
    )
    plot_confusion_matrix(
        keras_true,
        keras_pred,
        class_names,
        report_dir / "keras_confusion_matrix.png",
    )
    save_metrics(keras_metrics, report_dir / "keras_metrics.json")

    print("开始评估 TFLite 模型。")
    interpreter = tf.lite.Interpreter(
        model_path=str(tflite_model_path),
        num_threads=num_threads,
    )
    interpreter.allocate_tensors()
    tflite_true, tflite_pred, tflite_confidence = run_tflite_inference(interpreter, dataset)
    tflite_metrics = build_metrics(tflite_true, tflite_pred, class_names, source_name)
    save_classification_report(
        tflite_true,
        tflite_pred,
        class_names,
        report_dir / "tflite_classification_report.txt",
    )
    plot_confusion_matrix(
        tflite_true,
        tflite_pred,
        class_names,
        report_dir / "tflite_confusion_matrix.png",
    )
    save_metrics(tflite_metrics, report_dir / "tflite_metrics.json")

    if keras_true != tflite_true:
        raise ValueError("Keras 与 TFLite 评估样本顺序不一致，无法比较。")

    comparison = build_prediction_comparison(
        keras_true,
        keras_pred,
        tflite_pred,
        keras_metrics,
        tflite_metrics,
    )
    save_metrics(comparison, report_dir / "comparison_summary.json")
    save_disagreement_csv(
        keras_true,
        keras_pred,
        tflite_pred,
        keras_confidence,
        tflite_confidence,
        class_names,
        report_dir / "prediction_disagreements.csv",
    )
    plot_metric_comparison(
        keras_metrics,
        tflite_metrics,
        report_dir / "metric_comparison.png",
    )
    plot_per_class_accuracy_comparison(
        keras_true,
        keras_pred,
        tflite_pred,
        class_names,
        report_dir / "per_class_accuracy_comparison.png",
    )

    print(f"对比评估完成，评估数据源: {source_name}")
    print(f"Keras accuracy: {keras_metrics['accuracy']:.4f}")
    print(f"TFLite accuracy: {tflite_metrics['accuracy']:.4f}")
    print(f"Accuracy gap (Keras - TFLite): {comparison['accuracy_gap']:.4f}")
    print(f"结果已保存到: {report_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="对比 Keras 与 TFLite 模型评估结果")
    parser.add_argument(
        "--keras-model",
        type=Path,
        default=Path("artifacts/best_model.keras"),
        help="Keras 模型路径",
    )
    parser.add_argument(
        "--tflite-model",
        type=Path,
        default=Path("artifacts/model.tflite"),
        help="TFLite 模型路径",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="配置文件路径",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=None,
        help="对比结果输出目录，默认使用 config.yaml 中的 paths.compare_report_dir",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=None,
        help="类别名称 JSON 文件路径，默认读取 Keras 模型同目录下的 class_names.json",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="TFLite 推理线程数",
    )
    args = parser.parse_args()
    main(
        args.keras_model,
        args.tflite_model,
        args.config,
        args.output_dir,
        args.class_names,
        args.num_threads,
    )
