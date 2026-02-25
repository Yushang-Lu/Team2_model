import matplotlib.pyplot as plt
import numpy as np
import itertools
from sklearn.metrics import confusion_matrix, classification_report
import seaborn as sns

def plot_training_history(history, save_path):
    """绘制训练曲线（准确率和损失）"""
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))
    
    # 准确率
    ax1.plot(history['accuracy'], label='train_acc')
    if 'val_accuracy' in history:
        ax1.plot(history['val_accuracy'], label='val_acc')
    ax1.set_title('Accuracy')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Accuracy')
    ax1.legend()
    
    # 损失
    ax2.plot(history['loss'], label='train_loss')
    if 'val_loss' in history:
        ax2.plot(history['val_loss'], label='val_loss')
    ax2.set_title('Loss')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Loss')
    ax2.legend()
    
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def plot_confusion_matrix(y_true, y_pred, class_names, save_path):
    """绘制混淆矩阵热力图"""
    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=class_names, yticklabels=class_names)
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.title('Confusion Matrix')
    plt.tight_layout()
    plt.savefig(save_path)
    plt.close()

def save_classification_report(y_true, y_pred, class_names, save_path):
    """将分类报告保存为txt文件"""
    report = classification_report(y_true, y_pred, target_names=class_names)
    with open(save_path, 'w') as f:
        f.write(report)
    print(f"Classification report saved to {save_path}")