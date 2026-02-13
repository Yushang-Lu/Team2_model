import os
import yaml
import tensorflow as tf
import matplotlib.pyplot as plt
ImageDataGenerator = tf.keras.preprocessing.image.ImageDataGenerator

def load_config(config_path='config.yaml'):
    """加载YAML配置文件"""
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    return config

def create_data_generators(config):
    """
    从目录创建训练和验证数据生成器，并应用数据增强
    """
    # 训练数据增强
    train_datagen = ImageDataGenerator(
        rescale=1./255,
        rotation_range=config['augmentation']['rotation_range'],
        width_shift_range=config['augmentation']['width_shift_range'],
        height_shift_range=config['augmentation']['height_shift_range'],
        shear_range=config['augmentation']['shear_range'],
        zoom_range=config['augmentation']['zoom_range'],
        horizontal_flip=config['augmentation']['horizontal_flip'],
        fill_mode=config['augmentation']['fill_mode']
    )
    
    # 验证集仅缩放
    val_datagen = ImageDataGenerator(rescale=1./255)
    
    train_generator = train_datagen.flow_from_directory(
        config['data']['train_dir'],
        target_size=config['model']['input_size'][:2],
        batch_size=config['training']['batch_size'],
        class_mode='categorical',
        shuffle=True
    )
    
    val_generator = val_datagen.flow_from_directory(
        config['data']['val_dir'],
        target_size=config['model']['input_size'][:2],
        batch_size=config['training']['batch_size'],
        class_mode='categorical',
        shuffle=False
    )
    
    return train_generator, val_generator

def plot_training_history(history, save_path='outputs/training_curves.png'):
    """
    绘制训练过程的损失和准确率曲线
    """
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 损失曲线
    ax1.plot(history.history['loss'], label='train_loss')
    ax1.plot(history.history['val_loss'], label='val_loss')
    ax1.set_title('Loss Curves')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()
    ax1.grid(True)
    
    # 准确率曲线
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