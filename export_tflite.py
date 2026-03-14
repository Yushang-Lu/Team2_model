import os
import yaml
import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small # type: ignore
from tensorflow.keras import layers, Model # type: ignore

def build_model(config):
    input_shape = config['model']['input_shape']
    num_classes = config['data']['num_classes']

    base_model = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=None,
        pooling='avg'
    )

    base_model.trainable = not config['model'].get('freeze_backbone', True)

    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)

    return model

def main(weights_path, output_path, config_path):
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)

    model = build_model(config)
    model.load_weights(weights_path)

    # Run one inference to ensure variables/resources are initialized
    input_shape = config['model']['input_shape']
    dummy = tf.zeros([1] + list(input_shape), dtype=tf.float32)
    model(dummy, training=False)

    # Create a concrete function for conversion (more robust)
    spec = tf.TensorSpec([1] + list(input_shape), tf.float32)
    run_model = tf.function(lambda x: model(x, training=False))
    concrete_func = run_model.get_concrete_function(spec)

    converter = tf.lite.TFLiteConverter.from_concrete_functions([concrete_func])
    # Allow fallback to select TF ops if needed
    converter.target_spec.supported_ops = [
        tf.lite.OpsSet.TFLITE_BUILTINS,
        tf.lite.OpsSet.SELECT_TF_OPS,
    ]
    # Optional: enable optimizations / quantization
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]

    tflite_model = converter.convert()

    os.makedirs(os.path.dirname(output_path), exist_ok=True)
    with open(output_path, 'wb') as f:
        f.write(tflite_model)

    print(f"TFLite model saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Convert Keras .weights.h5 to TFLite')
    parser.add_argument('--weights', type=str, default='models/best_model.weights.h5',
                        help='Path to Keras weights file (.weights.h5)')
    parser.add_argument('--output', type=str, default='models/model.tflite',
                        help='Output TFLite file path')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Config file containing model architecture info')
    args = parser.parse_args()
    main(args.weights, args.output, args.config)