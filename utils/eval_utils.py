from __future__ import annotations

from pathlib import Path
from typing import Any

import numpy as np
import tensorflow as tf
import yaml
from sklearn.metrics import classification_report


def load_config(config_path: Path) -> dict[str, Any]:
    with config_path.open("r", encoding="utf-8") as file:
        return yaml.safe_load(file)


def resolve_class_names_path(model_path: Path, explicit_path: Path | None) -> Path:
    if explicit_path is not None:
        return explicit_path
    return model_path.parent / "class_names.json"


def quantize_input(sample: np.ndarray, input_details: dict[str, Any]) -> np.ndarray:
    dtype = input_details["dtype"]
    scale, zero_point = input_details["quantization"]

    if np.issubdtype(dtype, np.integer):
        if scale == 0:
            raise ValueError("TFLite 输入张量量化参数无效。")
        quantized = np.round(sample / scale + zero_point)
        info = np.iinfo(dtype)
        return np.clip(quantized, info.min, info.max).astype(dtype)

    return sample.astype(dtype)


def dequantize_output(output: np.ndarray, output_details: dict[str, Any]) -> np.ndarray:
    dtype = output_details["dtype"]
    scale, zero_point = output_details["quantization"]

    if np.issubdtype(dtype, np.integer) and scale > 0:
        return scale * (output.astype(np.float32) - zero_point)

    return output.astype(np.float32)


def run_keras_inference(
    model: tf.keras.Model,
    dataset: tf.data.Dataset,
) -> tuple[list[int], list[int], list[float]]:
    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []

    for batch_index, (images, labels) in enumerate(dataset, start=1):
        probabilities = model(images, training=False).numpy()
        predictions = np.argmax(probabilities, axis=1)

        y_true.extend(int(label) for label in labels.numpy())
        y_pred.extend(int(prediction) for prediction in predictions)
        confidences.extend(float(value) for value in np.max(probabilities, axis=1))

        if batch_index % 10 == 0:
            print(f"已完成 {batch_index} 个 batch 的 Keras 推理。")

    return y_true, y_pred, confidences


def run_tflite_inference(
    interpreter: tf.lite.Interpreter,
    dataset: tf.data.Dataset,
) -> tuple[list[int], list[int], list[float]]:
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    y_true: list[int] = []
    y_pred: list[int] = []
    confidences: list[float] = []

    for batch_index, (images, labels) in enumerate(dataset, start=1):
        image_batch = images.numpy()
        label_batch = labels.numpy()

        for image, label in zip(image_batch, label_batch):
            sample = np.expand_dims(image, axis=0)
            interpreter.set_tensor(
                input_details["index"],
                quantize_input(sample, input_details),
            )
            interpreter.invoke()
            output = interpreter.get_tensor(output_details["index"])
            probabilities = dequantize_output(output, output_details)[0]
            predicted_index = int(np.argmax(probabilities))

            y_true.append(int(label))
            y_pred.append(predicted_index)
            confidences.append(float(np.max(probabilities)))

        if batch_index % 10 == 0:
            print(f"已完成 {batch_index} 个 batch 的 TFLite 推理。")

    return y_true, y_pred, confidences


def build_metrics(
    y_true: list[int],
    y_pred: list[int],
    class_names: list[str],
    source_name: str,
) -> dict[str, Any]:
    report_dict = classification_report(
        y_true,
        y_pred,
        target_names=class_names,
        digits=4,
        output_dict=True,
        zero_division=0,
    )
    return {
        "source_split": source_name,
        "num_samples": len(y_true),
        "accuracy": float(report_dict["accuracy"]),
        "macro_avg": {
            "precision": float(report_dict["macro avg"]["precision"]),
            "recall": float(report_dict["macro avg"]["recall"]),
            "f1_score": float(report_dict["macro avg"]["f1-score"]),
        },
        "weighted_avg": {
            "precision": float(report_dict["weighted avg"]["precision"]),
            "recall": float(report_dict["weighted avg"]["recall"]),
            "f1_score": float(report_dict["weighted avg"]["f1-score"]),
        },
    }


def build_prediction_comparison(
    y_true: list[int],
    keras_pred: list[int],
    tflite_pred: list[int],
    keras_metrics: dict[str, Any],
    tflite_metrics: dict[str, Any],
) -> dict[str, Any]:
    if len(y_true) != len(keras_pred) or len(y_true) != len(tflite_pred):
        raise ValueError("预测结果长度不一致，无法进行对比。")

    y_true_array = np.array(y_true)
    keras_pred_array = np.array(keras_pred)
    tflite_pred_array = np.array(tflite_pred)

    keras_correct = keras_pred_array == y_true_array
    tflite_correct = tflite_pred_array == y_true_array
    agreement = keras_pred_array == tflite_pred_array

    return {
        "num_samples": int(len(y_true)),
        "agreement_rate": float(np.mean(agreement)),
        "disagreement_count": int(np.sum(~agreement)),
        "both_correct_count": int(np.sum(keras_correct & tflite_correct)),
        "keras_only_correct_count": int(np.sum(keras_correct & ~tflite_correct)),
        "tflite_only_correct_count": int(np.sum(~keras_correct & tflite_correct)),
        "both_wrong_count": int(np.sum(~keras_correct & ~tflite_correct)),
        "accuracy_gap": float(keras_metrics["accuracy"] - tflite_metrics["accuracy"]),
        "macro_f1_gap": float(
            keras_metrics["macro_avg"]["f1_score"]
            - tflite_metrics["macro_avg"]["f1_score"]
        ),
        "weighted_f1_gap": float(
            keras_metrics["weighted_avg"]["f1_score"]
            - tflite_metrics["weighted_avg"]["f1_score"]
        ),
    }
