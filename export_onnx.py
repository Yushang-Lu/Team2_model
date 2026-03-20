import os
import yaml
import argparse
import tensorflow as tf
from tensorflow.keras.applications import MobileNetV3Large # type: ignore
from tensorflow.keras import layers, Model # type: ignore
import tf2onnx

def build_model(config):
    """
    根据配置构建 MobileNetV3Large 迁移学习模型。
    返回完整的 Keras 模型（忽略基模型，只需主模型用于权重加载和转换）。
    """
    input_shape = config['model']['input_shape']
    num_classes = config['data']['num_classes']
    
    # 加载基模型（此处设置 weights=None 避免加载预训练权重，加快构建速度）
    # 随后加载 .weights.h5 会覆盖预训练权重。
    base_model = MobileNetV3Large(
        input_shape=input_shape,
        include_top=False,
        weights=None,                # 改为 None，仅构建架构
        pooling='avg'                # 全局平均池化
    )
    
    # 根据配置决定是否冻结基模型（转换时不影响推理）
    base_model.trainable = not config['model'].get('freeze_backbone', True)  # 默认冻结
    
    # 构建完整模型
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # 冻结时设置 training=False
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    
    return model   # 只需返回主模型

def main(weights_path, output_path, config_path):
    # 1. 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 2. 根据配置构建模型架构（必须与训练时完全一致）
    model = build_model(config)
    
    # 3. 加载权重文件（.weights.h5 仅包含权重）
    model.load_weights(weights_path)
    
    # 4. 准备输入签名（batch 维度动态）
    input_shape = config['model']['input_shape']
    input_signature = [tf.TensorSpec([None, *input_shape], tf.float32, name='input')]
    
    # 5. 转换为 ONNX
    onnx_model, _ = tf2onnx.convert.from_keras(
        model,
        input_signature=input_signature,
        output_path=output_path,
        opset=13  # 推荐 opset
    )
    
    print(f"ONNX model saved to {output_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--weights', type=str, default='models/best_model.weights.h5',
                        help='Path to Keras weights file (.weights.h5)')
    parser.add_argument('--output', type=str, default='models/model.onnx',
                        help='Output ONNX path')
    parser.add_argument('--config', type=str, default='config.yaml',
                        help='Config file containing model architecture info')
    args = parser.parse_args()
    main(args.weights, args.output, args.config)