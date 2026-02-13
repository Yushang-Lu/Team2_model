import tensorflow as tf
from tensorflow.python.keras import layers, Model

def build_model(input_shape=(64,64,3), num_classes=3):
    """
    构建深度可分离卷积神经网络，轻量化设计
    """
    inputs = layers.Input(shape=input_shape)
    
    # Block 1 (2层可分离卷积)
    x = layers.SeparableConv2D(32, (3,3), padding='same')(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(32, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)   # 32x32
    
    # Block 2 (2层)
    x = layers.SeparableConv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(64, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)   # 16x16
    
    # Block 3 (2层)
    x = layers.SeparableConv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(128, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)   # 8x8
    
    # Block 4 (2层)
    x = layers.SeparableConv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(256, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.MaxPooling2D((2,2))(x)   # 4x4
    
    # Block 5 (2层)
    x = layers.SeparableConv2D(512, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    x = layers.SeparableConv2D(512, (3,3), padding='same')(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 不池化，直接全局平均池化
    
    # 分类头
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    
    model = Model(inputs, outputs, name='Lightweight_SeparableCNN')
    return model

if __name__ == '__main__':
    # 快速测试模型结构
    model = build_model()
    model.summary()
    print(f"参数量: {model.count_params()}")