from __future__ import annotations

from typing import Any

import tensorflow as tf
from tensorflow.keras import Model, layers # type: ignore
from tensorflow.keras.applications import MobileNetV3Large, MobileNetV3Small # type: ignore

BACKBONE_BUILDERS = {
    "MobileNetV3Small": MobileNetV3Small,
    "MobileNetV3Large": MobileNetV3Large,
}
DEFAULT_BACKBONE = "MobileNetV3Small"


def build_classifier(config: dict[str, Any]) -> tuple[Model, tf.keras.Model]:
    """Build a MobileNetV3 transfer learning classifier."""
    model_cfg = config["model"]
    data_cfg = config["data"]

    backbone_name = model_cfg.get("backbone", DEFAULT_BACKBONE)
    if backbone_name not in BACKBONE_BUILDERS:
        raise ValueError(
            "当前版本仅支持以下 backbone: "
            f"{sorted(BACKBONE_BUILDERS)}，收到: {backbone_name}"
        )

    input_shape = tuple(data_cfg["image_size"]) + (3,)
    num_classes = int(data_cfg["num_classes"])
    dropout_rate = float(model_cfg.get("dropout_rate", 0.2))
    backbone_builder = BACKBONE_BUILDERS[backbone_name]

    # Keep MobileNetV3's built-in preprocessing enabled so the backbone
    # receives the same input scale as the ImageNet pretraining setup.
    base_model = backbone_builder(
        input_shape=input_shape,
        include_top=False,
        weights=model_cfg.get("weights", "imagenet"),
    )
    base_model.trainable = False

    inputs = tf.keras.Input(shape=input_shape, name="image")
    x = base_model(inputs, training=False)
    x = layers.GlobalAveragePooling2D(name="avg_pool")(x)
    x = layers.Dropout(dropout_rate, name="dropout")(x)
    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        name="predictions",
    )(x)

    model = Model(
        inputs=inputs,
        outputs=outputs,
        name=f"{backbone_name.lower()}_classifier",
    )
    return model, base_model


def set_backbone_trainable_layers(
    base_model: tf.keras.Model,
    fine_tune_layers: int,
) -> None:
    """Unfreeze the tail of the backbone while keeping BatchNorm frozen."""
    for layer in base_model.layers:
        layer.trainable = False

    if fine_tune_layers == 0:
        return

    if fine_tune_layers < 0:
        candidate_layers = list(base_model.layers)
    else:
        candidate_layers = list(base_model.layers[-fine_tune_layers:])

    for layer in candidate_layers:
        if isinstance(layer, tf.keras.layers.BatchNormalization):
            layer.trainable = False
        else:
            layer.trainable = True
