from __future__ import annotations

import argparse
from pathlib import Path

import tensorflow as tf

from utils.data_utils import create_evaluation_dataset, load_class_names
from utils.eval_utils import (
    build_metrics,
    load_config,
    resolve_class_names_path,
    run_tflite_inference,
)
from utils.visualize import (
    plot_confidence_histogram,
    plot_confusion_matrix,
    plot_per_class_accuracy,
    save_classification_report,
    save_metrics,
)


def resolve_output_dir(config: dict[str, object], explicit_dir: Path | None) -> Path:
    if explicit_dir is not None:
        return explicit_dir
    paths_cfg = config.get("paths", {})
    return Path(paths_cfg.get("tflite_report_dir", "reports/tflite"))


def main(
    model_path: Path,
    config_path: Path,
    output_dir: Path | None,
    class_names_path: Path | None,
    num_threads: int,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite 模型文件不存在: {model_path}")

    config = load_config(config_path)
    class_names = load_class_names(resolve_class_names_path(model_path, class_names_path))
    dataset, source_name = create_evaluation_dataset(config, class_names)
    report_dir = resolve_output_dir(config, output_dir)
    report_dir.mkdir(parents=True, exist_ok=True)

    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=num_threads,
    )
    interpreter.allocate_tensors()

    y_true, y_pred, confidences = run_tflite_inference(interpreter, dataset)
    save_classification_report(
        y_true,
        y_pred,
        class_names,
        report_dir / "classification_report.txt",
    )
    plot_confusion_matrix(y_true, y_pred, class_names, report_dir / "confusion_matrix.png")
    plot_per_class_accuracy(y_true, y_pred, class_names, report_dir / "per_class_accuracy.png")
    plot_confidence_histogram(confidences, report_dir / "confidence_histogram.png")
    save_metrics(
        build_metrics(y_true, y_pred, class_names, source_name),
        report_dir / "metrics.json",
    )

    print(f"TFLite 测试完成，评估数据源: {source_name}")
    print(f"测试样本数: {len(y_true)}")
    print(f"结果已保存到: {report_dir}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="评估 TFLite 模型并可视化结果")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/model.tflite"),
        help="待评估的 TFLite 模型路径",
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
        help="评估结果输出目录，默认使用 config.yaml 中的 paths.tflite_report_dir",
    )
    parser.add_argument(
        "--class-names",
        type=Path,
        default=None,
        help="类别名称 JSON 文件路径，默认读取模型同目录下的 class_names.json",
    )
    parser.add_argument(
        "--num-threads",
        type=int,
        default=1,
        help="TFLite 推理线程数",
    )
    args = parser.parse_args()
    main(
        args.model,
        args.config,
        args.output_dir,
        args.class_names,
        args.num_threads,
    )
