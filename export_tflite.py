from __future__ import annotations

import argparse
import shutil
from pathlib import Path
from typing import Any

import tensorflow as tf
import yaml
from tensorflow.python.framework.convert_to_constants import convert_variables_to_constants_v2


def build_converter(
    model: tf.keras.Model,
    input_shape: tuple[int, int, int],
) -> tf.lite.TFLiteConverter:
    dummy_input = tf.zeros([1, *input_shape], dtype=tf.float32)
    model(dummy_input, training=False)

    @tf.function(input_signature=[tf.TensorSpec([1, *input_shape], tf.float32, name="image")])
    def serving_fn(inputs: tf.Tensor) -> tf.Tensor:
        return model(inputs, training=False)

    concrete_function = serving_fn.get_concrete_function()
    frozen_function = convert_variables_to_constants_v2(concrete_function)
    return tf.lite.TFLiteConverter.from_concrete_functions(
        [frozen_function],
        model,
    )


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def main(
    model_path: Path,
    output_path: Path | None,
    quantize: str | None,
    config_path: Path,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")

    config = load_config(config_path)
    export_cfg = config.get("export", {})
    resolved_output = output_path or Path(export_cfg.get("tflite_path", "artifacts/model.tflite"))
    resolved_quantize = quantize or export_cfg.get("quantize", "dynamic")

    model = tf.keras.models.load_model(model_path)
    input_shape = model.input_shape[1:]
    if len(input_shape) != 3:
        raise ValueError(f"模型输入形状异常: {model.input_shape}")

    converter = build_converter(
        model,
        (int(input_shape[0]), int(input_shape[1]), int(input_shape[2])),
    )

    if resolved_quantize == "dynamic":
        converter.optimizations = [tf.lite.Optimize.DEFAULT]
    elif resolved_quantize != "none":
        raise ValueError(f"不支持的量化模式: {resolved_quantize}")

    tflite_model = converter.convert()
    resolved_output.parent.mkdir(parents=True, exist_ok=True)
    resolved_output.write_bytes(tflite_model)

    class_names_path = model_path.parent / "class_names.json"
    if class_names_path.exists():
        target_class_names_path = resolved_output.parent / "class_names.json"
        if class_names_path.resolve() != target_class_names_path.resolve():
            shutil.copy2(class_names_path, target_class_names_path)

    print(f"TFLite 模型已导出到: {resolved_output}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="导出 TFLite 模型")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/best_model.keras"),
        help="训练好的 .keras 模型路径",
    )
    parser.add_argument(
        "--output",
        type=Path,
        default=None,
        help="TFLite 输出路径，默认读取 config.yaml 中的 export.tflite_path",
    )
    parser.add_argument(
        "--quantize",
        type=str,
        default=None,
        choices=["dynamic", "none"],
        help="量化方式，默认读取 config.yaml 中的 export.quantize",
    )
    parser.add_argument(
        "--config",
        type=Path,
        default=Path("config.yaml"),
        help="配置文件路径",
    )
    args = parser.parse_args()
    main(args.model, args.output, args.quantize, args.config)
