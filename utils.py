import os
import yaml
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

def load_config(config_path='config.yaml'):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_data_generators(config):
    """
    从单一训练目录自动划分训练集与验证集，并应用数据增强
    """
    # 设置随机种子（若配置中存在）
    if 'seed' in config['data']:
        tf.random.set_seed(config['data']['seed'])
        np.random.seed(config['data']['seed'])

    # 训练数据增强（应用在训练子集上）
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=config['augmentation']['rotation_range'],
        width_shift_range=config['augmentation']['width_shift_range'],
        height_shift_range=config['augmentation']['height_shift_range'],
        shear_range=config['augmentation']['shear_range'],
        zoom_range=config['augmentation']['zoom_range'],
        horizontal_flip=config['augmentation']['horizontal_flip'],
        fill_mode=config['augmentation']['fill_mode'],
        validation_split=config['data']['validation_split']   # 用于划分
    )

    # 验证集只做归一化，不做增强（也使用 validation_split 从同一目录读取）
    val_datagen = ImageDataGenerator(
        rescale=1./255,
        validation_split=config['data']['validation_split']
    )

    train_generator = train_datagen.flow_from_directory(
        config['data']['train_dir'],
        target_size=config['model']['input_size'][:2],
        batch_size=config['training']['batch_size'],
        class_mode='categorical',
        subset='training',          # 训练子集
        shuffle=True,
        seed=config['data'].get('seed', None)
    )

    val_generator = val_datagen.flow_from_directory(
        config['data']['train_dir'],
        target_size=config['model']['input_size'][:2],
        batch_size=config['training']['batch_size'],
        class_mode='categorical',
        subset='validation',        # 验证子集
        shuffle=False,             # 验证集不需要打乱
        seed=config['data'].get('seed', None)
    )

    return train_generator, val_generator

def plot_training_history(history, save_path='outputs/training_curves.png'):
    """
    绘制训练过程的损失和准确率曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    ax1.plot(history.history['loss'], label='train_loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    ax2.plot(history.history['accuracy'], label='train_acc')
    ax2.plot(history.history['val_accuracy'], label='val_acc')
    ax2.set_title('Accuracy Curves')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy')
    ax2.legend()
    ax2.grid(True)
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"训练曲线已保存至 {save_path}")

def plot_confusion_matrix(model, val_generator, class_names, save_path='outputs/confusion_matrix.png'):
    """
    绘制最佳模型在验证集上的混淆矩阵
    """
    # 获取验证集所有真实标签
    y_true = val_generator.classes
    # 预测（返回概率，取 argmax）
    y_pred_probs = model.predict(val_generator, verbose=1)
    y_pred = np.argmax(y_pred_probs, axis=1)

    # 计算混淆矩阵
    cm = confusion_matrix(y_true, y_pred)
    
    # 绘制
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(cmap=plt.cm.Blues, ax=ax, values_format='d')
    plt.title('Confusion Matrix on Validation Set')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.show()
    print(f"混淆矩阵已保存至 {save_path}")