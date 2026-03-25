from __future__ import annotations

import json
from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight

AUTOTUNE = tf.data.AUTOTUNE
IMAGE_EXTENSIONS = {".png", ".jpg", ".jpeg"}


def create_datasets(
    config: dict[str, Any]
) -> tuple[tf.data.Dataset, tf.data.Dataset, list[str], dict[int, float]]:
    """Create training and validation datasets from directory structure."""
    data_cfg = config["data"]
    train_dir = Path(data_cfg["train_dir"])
    val_dir = Path(data_cfg["val_dir"]) if data_cfg.get("val_dir") else None
    image_size = tuple(data_cfg["image_size"])
    batch_size = int(data_cfg["batch_size"])
    expected_num_classes = int(data_cfg["num_classes"])
    seed = int(data_cfg.get("seed", 42))
    validation_split = float(data_cfg.get("validation_split", 0.15))

    if not train_dir.exists():
        raise FileNotFoundError(f"训练集目录不存在: {train_dir}")

    class_names = list_class_names(train_dir, expected_num_classes)

    if val_dir:
        if not val_dir.exists():
            raise FileNotFoundError(f"验证集目录不存在: {val_dir}")
        validate_class_names(val_dir, class_names, expected_num_classes)
        train_file_paths, train_labels = gather_labeled_files(train_dir, class_names)
        val_file_paths, val_labels = gather_labeled_files(val_dir, class_names)
    else:
        train_file_paths, train_labels, val_file_paths, val_labels = stratified_split_dataset(
            train_dir,
            class_names,
            validation_split,
            seed,
        )

    train_ds = build_dataset_from_paths(
        train_file_paths,
        train_labels,
        image_size,
        batch_size,
        shuffle=True,
        seed=seed,
    )
    val_ds = build_dataset_from_paths(
        val_file_paths,
        val_labels,
        image_size,
        batch_size,
        shuffle=False,
        seed=seed,
    )
    class_weights = compute_class_weights_from_labels(train_labels, class_names)

    augmentation_cfg = data_cfg.get("augmentation", {})
    train_ds = prepare_dataset(
        train_ds,
        training=True,
        augmentation_cfg=augmentation_cfg,
        cache=False,
    )
    val_ds = prepare_dataset(val_ds, training=False, cache=True)
    return train_ds, val_ds, class_names, class_weights


def create_evaluation_dataset(
    config: dict[str, Any],
    class_names: list[str],
) -> tuple[tf.data.Dataset, str]:
    """Create evaluation dataset for exported TFLite model."""
    data_cfg = config["data"]
    train_dir = Path(data_cfg["train_dir"])
    val_dir = Path(data_cfg["val_dir"]) if data_cfg.get("val_dir") else None
    test_dir = Path(data_cfg["test_dir"]) if data_cfg.get("test_dir") else None
    image_size = tuple(data_cfg["image_size"])
    batch_size = int(data_cfg["batch_size"])
    expected_num_classes = int(data_cfg["num_classes"])
    seed = int(data_cfg.get("seed", 42))
    validation_split = float(data_cfg.get("validation_split", 0.15))

    if test_dir:
        if not test_dir.exists():
            raise FileNotFoundError(f"测试集目录不存在: {test_dir}")
        validate_class_names(test_dir, class_names, expected_num_classes)
        eval_file_paths, eval_labels = gather_labeled_files(test_dir, class_names)
        source_name = "test_dir"
    elif val_dir:
        if not val_dir.exists():
            raise FileNotFoundError(f"验证集目录不存在: {val_dir}")
        validate_class_names(val_dir, class_names, expected_num_classes)
        eval_file_paths, eval_labels = gather_labeled_files(val_dir, class_names)
        source_name = "val_dir"
    else:
        if not train_dir.exists():
            raise FileNotFoundError(f"训练集目录不存在: {train_dir}")
        _, _, eval_file_paths, eval_labels = stratified_split_dataset(
            train_dir,
            class_names,
            validation_split,
            seed,
        )
        source_name = "validation_split"

    eval_ds = build_dataset_from_paths(
        eval_file_paths,
        eval_labels,
        image_size,
        batch_size,
        shuffle=False,
        seed=seed,
    )
    eval_ds = prepare_dataset(eval_ds, training=False, cache=True)
    return eval_ds, source_name


def build_dataset_from_paths(
    file_paths: list[str],
    labels: list[int],
    image_size: tuple[int, int],
    batch_size: int,
    shuffle: bool,
    seed: int,
) -> tf.data.Dataset:
    if not file_paths:
        raise ValueError("数据集文件列表为空，无法构建数据集。")
    if len(file_paths) != len(labels):
        raise ValueError("文件路径和标签数量不一致。")

    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    if shuffle:
        dataset = dataset.shuffle(
            buffer_size=len(file_paths),
            seed=seed,
            reshuffle_each_iteration=True,
        )
    dataset = dataset.map(
        lambda file_path, label: decode_image(file_path, label, image_size),
        num_parallel_calls=AUTOTUNE,
    )
    return dataset.batch(batch_size)


def decode_image(
    file_path: tf.Tensor,
    label: tf.Tensor,
    image_size: tuple[int, int],
) -> tuple[tf.Tensor, tf.Tensor]:
    image_bytes = tf.io.read_file(file_path)
    image = tf.io.decode_image(image_bytes, channels=3, expand_animations=False)
    image.set_shape([None, None, 3])
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)
    return image, label


def prepare_dataset(
    dataset: tf.data.Dataset,
    training: bool,
    augmentation_cfg: dict[str, Any] | None = None,
    cache: bool = False,
) -> tf.data.Dataset:
    augmentation_cfg = augmentation_cfg or {}
    augmenter = build_augmenter(augmentation_cfg)
    brightness_delta = float(augmentation_cfg.get("brightness_delta", 0.0))
    contrast_factor = float(augmentation_cfg.get("contrast_factor", 0.0))

    def preprocess_train(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        images = tf.cast(images, tf.float32)
        if augmenter is not None:
            images = augmenter(images, training=True)
        if brightness_delta > 0 or contrast_factor > 0:
            images = images / 255.0
            if brightness_delta > 0:
                images = tf.image.random_brightness(images, max_delta=brightness_delta)
            if contrast_factor > 0:
                images = tf.image.random_contrast(
                    images,
                    lower=max(0.0, 1.0 - contrast_factor),
                    upper=1.0 + contrast_factor,
                )
            images = tf.clip_by_value(images, 0.0, 1.0) * 255.0
        return images, labels

    def preprocess_eval(images: tf.Tensor, labels: tf.Tensor) -> tuple[tf.Tensor, tf.Tensor]:
        images = tf.cast(images, tf.float32)
        return images, labels

    preprocess_fn = preprocess_train if training else preprocess_eval
    dataset = dataset.map(preprocess_fn, num_parallel_calls=AUTOTUNE)
    if cache:
        dataset = dataset.cache()
    return dataset.prefetch(AUTOTUNE)


def build_augmenter(augmentation_cfg: dict[str, Any]) -> tf.keras.Sequential | None:
    layers: list[tf.keras.layers.Layer] = []

    if augmentation_cfg.get("horizontal_flip", True):
        layers.append(tf.keras.layers.RandomFlip("horizontal"))

    rotation_factor = float(augmentation_cfg.get("rotation_factor", 0.03))
    if rotation_factor > 0:
        layers.append(tf.keras.layers.RandomRotation(rotation_factor))

    zoom_factor = float(augmentation_cfg.get("zoom_factor", 0.08))
    if zoom_factor > 0:
        layers.append(
            tf.keras.layers.RandomZoom(
                height_factor=zoom_factor,
                width_factor=zoom_factor,
            )
        )

    if not layers:
        return None

    return tf.keras.Sequential(layers, name="light_augmentation")


def list_class_names(data_dir: Path, expected_num_classes: int) -> list[str]:
    class_names = sorted(
        path.name for path in data_dir.iterdir() if path.is_dir() and not path.name.startswith(".")
    )
    if len(class_names) != expected_num_classes:
        raise ValueError(
            f"期望 {expected_num_classes} 个类别目录，但在 {data_dir} 中找到 {len(class_names)} 个: {class_names}"
        )
    return class_names


def validate_class_names(
    data_dir: Path,
    class_names: list[str],
    expected_num_classes: int,
) -> None:
    observed = list_class_names(data_dir, expected_num_classes)
    if observed != class_names:
        raise ValueError(f"类别目录与训练集不一致。train={class_names}, observed={observed}")


def gather_labeled_files(data_dir: Path, class_names: list[str]) -> tuple[list[str], list[int]]:
    file_paths: list[str] = []
    labels: list[int] = []
    for class_index, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        for image_path in sorted(class_dir.rglob("*")):
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS:
                file_paths.append(str(image_path))
                labels.append(class_index)

    if not file_paths:
        raise ValueError(f"在 {data_dir} 中没有找到可用图片文件。支持格式: {sorted(IMAGE_EXTENSIONS)}")

    return file_paths, labels


def stratified_split_dataset(
    data_dir: Path,
    class_names: list[str],
    validation_split: float,
    seed: int,
) -> tuple[list[str], list[int], list[str], list[int]]:
    if not 0.0 < validation_split < 1.0:
        raise ValueError(f"validation_split 必须在 (0, 1) 之间，收到: {validation_split}")

    rng = np.random.default_rng(seed)
    train_file_paths: list[str] = []
    train_labels: list[int] = []
    val_file_paths: list[str] = []
    val_labels: list[int] = []

    for class_index, class_name in enumerate(class_names):
        class_dir = data_dir / class_name
        class_file_paths = [
            str(image_path)
            for image_path in sorted(class_dir.rglob("*"))
            if image_path.is_file() and image_path.suffix.lower() in IMAGE_EXTENSIONS
        ]

        if not class_file_paths:
            raise ValueError(f"类别 {class_name} 中没有可用图片，无法切分。")
        if len(class_file_paths) < 2:
            raise ValueError(
                f"类别 {class_name} 只有 {len(class_file_paths)} 张图片，"
                "无法在没有独立验证集时进行分层切分。"
            )

        shuffled_indices = rng.permutation(len(class_file_paths))
        num_val = int(round(len(class_file_paths) * validation_split))
        num_val = min(max(num_val, 1), len(class_file_paths) - 1)

        val_class_paths = sorted(class_file_paths[index] for index in shuffled_indices[:num_val])
        train_class_paths = sorted(class_file_paths[index] for index in shuffled_indices[num_val:])

        train_file_paths.extend(train_class_paths)
        train_labels.extend([class_index] * len(train_class_paths))
        val_file_paths.extend(val_class_paths)
        val_labels.extend([class_index] * len(val_class_paths))

    return train_file_paths, train_labels, val_file_paths, val_labels


def compute_class_weights(data_dir: Path, class_names: list[str]) -> dict[int, float]:
    _, labels = gather_labeled_files(data_dir, class_names)
    return compute_class_weights_from_labels(labels, class_names)


def compute_class_weights_from_paths(
    file_paths: list[str],
    class_names: list[str],
) -> dict[int, float]:
    if not file_paths:
        raise ValueError("训练文件列表为空，无法计算类别权重。")

    class_to_index = {class_name: index for index, class_name in enumerate(class_names)}
    labels = [class_to_index[Path(file_path).parent.name] for file_path in file_paths]
    return compute_class_weights_from_labels(labels, class_names)


def compute_class_weights_from_labels(
    labels: list[int],
    class_names: list[str],
) -> dict[int, float]:
    classes = np.arange(len(class_names))
    weights = compute_class_weight(class_weight="balanced", classes=classes, y=np.array(labels))
    return {int(index): float(weight) for index, weight in zip(classes, weights)}


def save_class_names(class_names: list[str], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(
        json.dumps(class_names, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )


def load_class_names(path: Path) -> list[str]:
    if not path.exists():
        raise FileNotFoundError(f"类别名称文件不存在: {path}")
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, list) or not all(isinstance(item, str) for item in data):
        raise ValueError(f"类别名称文件格式错误: {path}")
    return data
