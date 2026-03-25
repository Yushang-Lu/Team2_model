from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import Model, layers # type: ignore

from .base import ModelBuildResult


def conv_bn_act(
    x: tf.Tensor,
    filters: int,
    kernel_size: int,
    strides: int,
    name: str,
) -> tf.Tensor:
    x = layers.Conv2D(
        filters,
        kernel_size,
        strides=strides,
        padding="same",
        use_bias=False,
        name=f"{name}_conv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    return layers.Activation("swish", name=f"{name}_act")(x)


def separable_block(
    x: tf.Tensor,
    filters: int,
    strides: int,
    name: str,
) -> tf.Tensor:
    residual = x
    x = layers.SeparableConv2D(
        filters,
        kernel_size=3,
        strides=strides,
        padding="same",
        use_bias=False,
        depthwise_initializer="he_normal",
        pointwise_initializer="he_normal",
        name=f"{name}_sepconv",
    )(x)
    x = layers.BatchNormalization(name=f"{name}_bn")(x)
    x = layers.Activation("swish", name=f"{name}_act")(x)

    if strides == 1 and residual.shape[-1] == x.shape[-1]:
        x = layers.Add(name=f"{name}_add")([x, residual])

    return x


def build_classifier(config: dict[str, Any]) -> ModelBuildResult:
    data_cfg = config["data"]
    model_cfg = config["model"]

    input_shape = tuple(data_cfg["image_size"]) + (3,)
    num_classes = int(data_cfg["num_classes"])
    dropout_rate = float(model_cfg.get("dropout_rate", 0.2))

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = layers.Rescaling(1.0 / 255.0, name="rescale")(inputs)
    x = conv_bn_act(x, filters=24, kernel_size=3, strides=2, name="stem")

    block_settings = [
        (24, 1),
        (48, 2),
        (64, 1),
        (96, 2),
        (128, 1),
        (160, 2),
    ]
    for block_index, (filters, strides) in enumerate(block_settings, start=1):
        x = separable_block(
            x,
            filters=filters,
            strides=strides,
            name=f"block{block_index}",
        )

    x = conv_bn_act(x, filters=192, kernel_size=1, strides=1, name="head")
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(num_classes, activation="softmax", name="predictions")(x)

    model = Model(inputs=inputs, outputs=outputs, name="lightweight_conv_classifier")
    return ModelBuildResult(
        model=model,
        fine_tune_target=None,
        supports_stage2=False,
    )
