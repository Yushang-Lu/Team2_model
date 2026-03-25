from __future__ import annotations

import argparse
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from PIL import Image

from utils.data_utils import load_class_names
from utils.eval_utils import dequantize_output, quantize_input, resolve_class_names_path


def load_image(image_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(target_size, resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32)
    return np.expand_dims(image_array, axis=0)


def resolve_input_size(input_details: dict[str, Any]) -> tuple[int, int]:
    input_shape = input_details["shape"]
    input_signature = input_details.get("shape_signature", input_shape)

    if len(input_shape) != 4:
        raise ValueError(f"TFLite 模型输入形状异常: {input_shape}")

    height = int(input_signature[1]) if int(input_signature[1]) > 0 else int(input_shape[1])
    width = int(input_signature[2]) if int(input_signature[2]) > 0 else int(input_shape[2])
    channels = int(input_signature[3]) if int(input_signature[3]) > 0 else int(input_shape[3])

    if height <= 0 or width <= 0:
        raise ValueError(f"无法解析 TFLite 模型输入尺寸: {input_shape}")
    if channels != 3:
        raise ValueError(f"当前仅支持 3 通道 RGB 输入，收到: {input_shape}")

    return height, width


def main(
    model_path: Path,
    image_path: Path,
    class_names_path: Path | None,
    num_threads: int,
) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"TFLite 模型文件不存在: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")
    if num_threads < 1:
        raise ValueError(f"num_threads 必须大于等于 1，收到: {num_threads}")

    class_names = load_class_names(resolve_class_names_path(model_path, class_names_path))
    interpreter = tf.lite.Interpreter(
        model_path=str(model_path),
        num_threads=num_threads,
    )
    interpreter.allocate_tensors()

    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
    image_batch = load_image(image_path, resolve_input_size(input_details))

    interpreter.set_tensor(
        input_details["index"],
        quantize_input(image_batch, input_details),
    )
    interpreter.invoke()
    probabilities = np.squeeze(
        dequantize_output(
            interpreter.get_tensor(output_details["index"]),
            output_details,
        )
    )
    if probabilities.ndim != 1:
        raise ValueError(f"TFLite 模型输出形状异常: {probabilities.shape}")
    if len(probabilities) != len(class_names):
        raise ValueError(
            "模型输出类别数与 class_names.json 不一致，"
            f"模型输出为 {len(probabilities)} 类，类别名称为 {len(class_names)} 类。"
        )
    predicted_index = int(np.argmax(probabilities))

    print(f"预测类别: {class_names[predicted_index]}")
    print("类别概率:")
    for class_name, probability in zip(class_names, probabilities):
        print(f"  {class_name}: {probability:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="使用 TFLite 模型进行单张图片预测")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/model.tflite"),
        help="训练好的 .tflite 模型路径",
    )
    parser.add_argument(
        "--image",
        type=Path,
        required=True,
        help="待预测图片路径",
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
    main(args.model, args.image, args.class_names, args.num_threads)
