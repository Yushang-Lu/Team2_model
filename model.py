import tensorflow as tf
from tensorflow.keras import layers, Model  # type: ignore

# 定义 L2 正则化器，系数可在此调整
# l2_reg = tf.keras.regularizers.l2(1e-3)
l2_reg = None

def inverted_residual_block(x, expand_ratio, filters, stride):
    shortcut = x
    in_channels = int(x.shape[-1])
    # 1x1 升维
    x = layers.Conv2D(in_channels * expand_ratio, (1, 1), padding='same',
                      kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 3x3 深度卷积
    x = layers.DepthwiseConv2D((3, 3), strides=stride, padding='same',
                               kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)
    # 1x1 降维
    x = layers.Conv2D(filters, (1, 1), padding='same',
                      kernel_regularizer=l2_reg)(x)
    x = layers.BatchNormalization()(x)
    # 残差连接：只有当步长为1且通道匹配时才加
    if stride == 1 and int(shortcut.shape[-1]) == filters:
        x = layers.Add()([x, shortcut])
    return x

def build_model(input_shape=(64, 64, 3), num_classes=3):
    """
    构建深度可分离卷积神经网络，轻量化设计
    """ 
    inputs = layers.Input(shape=input_shape)
    
    # 标准卷积
    x = layers.Conv2D(32, (3, 3), strides=2, padding='same',
                      kernel_regularizer=l2_reg)(inputs)
    x = layers.BatchNormalization()(x)
    x = layers.ReLU()(x)

    # 倒残差块堆叠
    x = inverted_residual_block(x, expand_ratio=1, filters=16, stride=1)  # 第一块可以不升维
    x = inverted_residual_block(x, expand_ratio=6, filters=24, stride=2)  # 下采样到16x16
    x = inverted_residual_block(x, expand_ratio=6, filters=24, stride=1)
    x = inverted_residual_block(x, expand_ratio=6, filters=32, stride=2)  # 下采样到8x8
    x = inverted_residual_block(x, expand_ratio=6, filters=32, stride=1)
    x = inverted_residual_block(x, expand_ratio=6, filters=64, stride=1)  # 保持8x8
    # 可选再增加一个块
    x = inverted_residual_block(x, expand_ratio=6, filters=64, stride=1)
    
    # 分类头
    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)
    outputs = layers.Dense(num_classes, activation='softmax',
                           kernel_regularizer=l2_reg)(x)
    
    model = Model(inputs, outputs, name='Lightweight_Model')
    return model

if __name__ == '__main__':
    # 快速测试模型结构
    model = build_model()
    model.summary()
    print(f"参数量: {model.count_params()}")