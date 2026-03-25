from __future__ import annotations

from dataclasses import dataclass

import tensorflow as tf


@dataclass(frozen=True)
class ModelBuildResult:
    model: tf.keras.Model
    fine_tune_target: tf.keras.Model | None
    supports_stage2: bool
