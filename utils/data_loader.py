import os
import numpy as np
import tensorflow as tf
from tensorflow.keras.applications.mobilenet_v3 import preprocess_input # type: ignore
from sklearn.utils.class_weight import compute_class_weight

AUTOTUNE = tf.data.AUTOTUNE

def get_class_names_from_dir(data_dir):
    """从子目录名获取类别名称（按字母排序确保一致）"""
    class_names = sorted([d for d in os.listdir(data_dir) if os.path.isdir(os.path.join(data_dir, d))])
    return class_names

def decode_image(filename, label, image_size):
    """读取、解码、resize图像,返回浮点型张量(范围[0,255])"""
    image = tf.io.read_file(filename)
    image = tf.image.decode_png(image, channels=3)  # 彩色图像
    image = tf.image.resize(image, image_size)
    image = tf.cast(image, tf.float32)  # 保持[0,255]以便后续preprocess_input
    return image, label

def create_augmentation_function(aug_cfg):
    """
    根据配置预创建所有的增强层，并返回一个可以应用于 (image, label) 对的函数。
    """
    # --- 预创建所有层 (只创建一次) ---
    rotation_layer = None
    rotation_deg = aug_cfg.get('random_rotation', 0)
    if rotation_deg > 0:
        rotation_layer = tf.keras.layers.RandomRotation(
            factor=rotation_deg / 360.0,  # 转换为比例
            fill_mode='nearest',
            interpolation='bilinear'
        )

    # 其他增强配置参数
    do_flip = aug_cfg.get('random_flip') == 'horizontal'
    brightness_delta = aug_cfg.get('random_brightness', 0)
    contrast_range = aug_cfg.get('random_contrast')

    # --- 定义实际的增强函数 (会被 map 调用) ---
    def augment(image, label):
        # 随机水平翻转 (TensorFlow 内置函数没有变量问题)
        if do_flip:
            image = tf.image.random_flip_left_right(image)

        # 应用预创建的旋转层
        if rotation_layer is not None:
            # RandomRotation 需要 batch 维度
            image = tf.expand_dims(image, axis=0)
            image = rotation_layer(image)
            image = tf.squeeze(image, axis=0)

        # 随机亮度
        if brightness_delta > 0:
            image = tf.image.random_brightness(image, max_delta=brightness_delta)

        # 随机对比度
        if contrast_range is not None:
            lower, upper = contrast_range
            image = tf.image.random_contrast(image, lower=lower, upper=upper)

        # 限制像素值范围
        image = tf.clip_by_value(image, 0.0, 255.0)
        return image, label

    return augment

def normalize_image(image, label):
    """使用MobileNetV3的预处理函数归一化到[-1,1]"""
    image = preprocess_input(image)  # 输入需为float32，范围[0,255]
    return image, label

def create_dataset(data_dir, image_size, batch_size, is_training=True, validation_split=None, augment_config=None):
    """
    从目录创建数据集（支持训练集/验证集）
    若validation_split不为None且is_training=True,则从训练集划分验证集
    """
    class_names = get_class_names_from_dir(data_dir)
    class_to_idx = {name: i for i, name in enumerate(class_names)}
    
    # 获取所有图像路径和标签
    file_paths = []
    labels = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        for fname in os.listdir(class_dir):
            if fname.lower().endswith('.png'):
                file_paths.append(os.path.join(class_dir, fname))
                labels.append(class_to_idx[class_name])
    
    # 转换为TensorFlow数据集
    dataset = tf.data.Dataset.from_tensor_slices((file_paths, labels))
    
    # 划分训练/验证
    if validation_split and is_training:
        num_val = int(len(file_paths) * validation_split)
        num_train = len(file_paths) - num_val
        # 随机打乱
        dataset = dataset.shuffle(buffer_size=len(file_paths), seed=42)
        train_dataset = dataset.take(num_train)
        val_dataset = dataset.skip(num_train)
    else:
        # 若没有划分，全部作为训练集（此时is_training决定是否打乱）
        train_dataset = dataset
        val_dataset = None
    
    # --- 预创建增强函数 ---
    augmentation_func = None
    if augment_config and is_training:
        augmentation_func = create_augmentation_function(augment_config['data']['augmentations'])

    # 定义处理函数
    def process_train(img, lbl):
        img, lbl = decode_image(img, lbl, image_size)
        if augmentation_func:  # 使用预创建的增强函数
            img, lbl = augmentation_func(img, lbl)
        img, lbl = normalize_image(img, lbl)
        return img, lbl
    
    def process_val(img, lbl):
        img, lbl = decode_image(img, lbl, image_size)
        img, lbl = normalize_image(img, lbl)
        return img, lbl
    
    # 构建训练集
    train_dataset = train_dataset.map(process_train, num_parallel_calls=AUTOTUNE)
    if is_training:
        train_dataset = train_dataset.shuffle(buffer_size=1000).repeat()
    train_dataset = train_dataset.batch(batch_size).prefetch(AUTOTUNE)
    
    # 构建验证集
    if val_dataset is not None:
        val_dataset = val_dataset.map(process_val, num_parallel_calls=AUTOTUNE)
        val_dataset = val_dataset.batch(batch_size).prefetch(AUTOTUNE)
    
    return train_dataset, val_dataset, class_names

def compute_class_weights(data_dir, class_names):
    """根据训练集样本数计算类别权重"""
    class_counts = []
    for class_name in class_names:
        class_dir = os.path.join(data_dir, class_name)
        count = len([f for f in os.listdir(class_dir) if f.endswith('.png')])
        class_counts.append(count)
    class_weights = compute_class_weight('balanced', classes=np.arange(len(class_names)), y=np.repeat(np.arange(len(class_names)), class_counts))
    return dict(enumerate(class_weights))