import tensorflow as tf
from tensorflow.keras import layers, Model # type: ignore
from tensorflow.keras.applications import MobileNetV3Small # type: ignore

def build_model(config):
    """构建MobileNetV3Small迁移学习模型"""
    input_shape = config['model']['input_shape']
    num_classes = config['data']['num_classes']
    
    # 加载基模型
    base_model = MobileNetV3Small(
        input_shape=input_shape,
        include_top=False,
        weights=config['model']['weights'],
        pooling='avg'  # 全局平均池化
    )
    
    # 初始冻结基模型
    base_model.trainable = not config['model']['freeze_backbone']
    
    # 构建完整模型
    inputs = tf.keras.Input(shape=input_shape)
    x = base_model(inputs, training=False)  # 冻结时设置training=False
    x = layers.Dropout(0.2)(x)
    outputs = layers.Dense(num_classes, activation='softmax')(x)
    model = Model(inputs, outputs)
    
    return model, base_model

def get_optimizer(optimizer_name, learning_rate):
    """根据名称返回优化器实例"""
    if optimizer_name.lower() == 'adam':
        return tf.keras.optimizers.Adam(learning_rate=learning_rate)
    elif optimizer_name.lower() == 'sgd':
        return tf.keras.optimizers.SGD(learning_rate=learning_rate, momentum=0.9)
    else:
        raise ValueError(f"Unsupported optimizer: {optimizer_name}")

def get_lr_scheduler(schedule_name, initial_lr, steps_per_epoch, total_epochs):
    """返回学习率调度器回调或schedule函数"""
    if schedule_name == 'cosine':
        # 使用余弦退火
        decay_steps = steps_per_epoch * total_epochs
        lr_schedule = tf.keras.optimizers.schedules.CosineDecay(
            initial_learning_rate=initial_lr,
            decay_steps=decay_steps,
            alpha=0.0  # 最终学习率降到0
        )
        return lr_schedule
    elif schedule_name == 'reduce_on_plateau':
        # ReduceLROnPlateau作为回调，这里返回None，由回调处理
        return None
    else:
        return None  # 恒定学习率