import os
import yaml
import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Small # type: ignore
from tensorflow.keras import layers, Model # type: ignore

def build_model(config):
    """
    根据配置构建 MobileNetV3Small 迁移学习模型。
    返回完整的 Keras 模型（用于加载权重和转换）。
    """
    input_shape = config['model']['input_shape']
    num_classes = config['data']['num_classes']
    
    # 加载基模型，仅构建架构，不加载预训练权重
    base_model = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=None,
        pooling='avg'
    )
    
    # 根据配置决定是否冻结基模型（不影响转换）
    base_model.trainable = not config['model'].get('freeze_backbone', True)
    
    # 构建完整模型
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # 推理模式
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    
    return model

def main(weights_path, output_path, config_path):
    # 1. 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. 构建模型架构
    model = build_model(config)
    
    # 3. 加载权重文件
    model.load_weights(weights_path)
    
    # 4. 转换为 TFLite 模型
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    # 可选：添加优化选项（如需要量化可取消下行注释）
    # converter.optimizations = [tf.lite.Optimize.DEFAULT]
    tflite_model = converter.convert()
    
    # 5. 保存 TFLite 模型
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