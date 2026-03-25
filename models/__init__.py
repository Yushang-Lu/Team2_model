from __future__ import annotations

from typing import Any, Callable

from .base import ModelBuildResult
from .conv_builder import build_classifier as build_conv_classifier
from .model_builder import (
    build_classifier as build_mobilenet_classifier,
    set_backbone_trainable_layers,
)

BuilderFunction = Callable[[dict[str, Any]], ModelBuildResult]

MODEL_BUILDERS: dict[str, BuilderFunction] = {
    "mobilenet": build_mobilenet_classifier,
    "conv": build_conv_classifier,
}
DEFAULT_BUILDER = "mobilenet"


def resolve_builder_name(config: dict[str, Any]) -> str:
    model_cfg = config.get("model", {})
    builder_name = str(model_cfg.get("builder", DEFAULT_BUILDER)).strip().lower()
    if builder_name not in MODEL_BUILDERS:
        raise ValueError(
            "当前版本仅支持以下 model.builder: "
            f"{sorted(MODEL_BUILDERS)}，收到: {builder_name}"
        )
    return builder_name


def build_classifier(config: dict[str, Any]) -> ModelBuildResult:
    builder_name = resolve_builder_name(config)
    return MODEL_BUILDERS[builder_name](config)


def prepare_stage2_training(
    build_result: ModelBuildResult,
    fine_tune_layers: int,
) -> bool:
    if not build_result.supports_stage2 or build_result.fine_tune_target is None:
        return False

    set_backbone_trainable_layers(build_result.fine_tune_target, fine_tune_layers)
    return True


__all__ = [
    "DEFAULT_BUILDER",
    "MODEL_BUILDERS",
    "ModelBuildResult",
    "build_classifier",
    "prepare_stage2_training",
    "resolve_builder_name",
]
