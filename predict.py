from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np
import tensorflow as tf
from PIL import Image
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input # type: ignore

from utils.data_utils import load_class_names


def resolve_class_names_path(model_path: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    return model_path.parent / "class_names.json"


def load_image(image_path: Path, target_size: tuple[int, int]) -> np.ndarray:
    resampling = getattr(Image, "Resampling", Image)
    with Image.open(image_path) as image:
        image = image.convert("RGB")
        image = image.resize(target_size, resampling.BILINEAR)
        image_array = np.asarray(image, dtype=np.float32)
    image_array = preprocess_input(image_array)
    return np.expand_dims(image_array, axis=0)


def main(model_path: Path, image_path: Path, class_names_path: Path | None) -> None:
    if not model_path.exists():
        raise FileNotFoundError(f"模型文件不存在: {model_path}")
    if not image_path.exists():
        raise FileNotFoundError(f"图片文件不存在: {image_path}")

    class_names = load_class_names(resolve_class_names_path(model_path, class_names_path))
    model = tf.keras.models.load_model(model_path)

    input_shape = model.input_shape
    if len(input_shape) != 4:
        raise ValueError(f"模型输入形状异常: {input_shape}")

    image_batch = load_image(image_path, (int(input_shape[1]), int(input_shape[2])))
    probabilities = model.predict(image_batch, verbose=0)[0]
    predicted_index = int(np.argmax(probabilities))

    print(f"预测类别: {class_names[predicted_index]}")
    print("类别概率:")
    for class_name, probability in zip(class_names, probabilities):
        print(f"  {class_name}: {probability:.6f}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="单张图片预测脚本")
    parser.add_argument(
        "--model",
        type=Path,
        default=Path("artifacts/best_model.keras"),
        help="训练好的 .keras 模型路径",
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
    args = parser.parse_args()
    main(args.model, args.image, args.class_names)
