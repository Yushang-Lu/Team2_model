import os
import yaml
import argparse
import numpy as np
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, CSVLogger, LearningRateScheduler # type: ignore
from utils.data_loader import create_dataset, compute_class_weights, get_class_names_from_dir
from models.model_builder import build_model, get_optimizer, get_lr_scheduler
from utils.visualize import plot_training_history, plot_confusion_matrix, save_classification_report

def main(config_path):
    # 加载配置
    with open(config_path, 'r') as f:
        config = yaml.safe_load(f)
    
    # 创建输出目录
    os.makedirs(config['logging']['log_dir'], exist_ok=True)
    os.makedirs(config['logging']['model_dir'], exist_ok=True)
    os.makedirs(config['logging']['output_dir'], exist_ok=True)
    
    # 数据加载
    train_dir = config['data']['train_dir']
    val_dir = config['data']['val_dir']
    image_size = tuple(config['data']['image_size'])
    batch_size = config['data']['batch_size']
    
    if val_dir and os.path.exists(val_dir):
        # 使用独立验证集
        train_dataset, _, class_names = create_dataset(
            train_dir, image_size, batch_size, is_training=True,
            augment_config=config, validation_split=None
        )
        val_dataset, _, _ = create_dataset(
            val_dir, image_size, batch_size, is_training=False,
            augment_config=None, validation_split=None
        )
    else:
        # 从训练集划分验证集
        validation_split = config['data']['validation_split']
        train_dataset, val_dataset, class_names = create_dataset(
            train_dir, image_size, batch_size, is_training=True,
            augment_config=config, validation_split=validation_split
        )
    
    # 计算类别权重
    class_weights = compute_class_weights(train_dir, class_names)
    print("Class weights:", class_weights)
    
    # 计算步数
    num_train_images = sum([len(files) for r, d, files in os.walk(train_dir) if any(f.endswith('.png') for f in files)])
    steps_per_epoch = int(num_train_images // batch_size)
    if val_dataset is not None:
        if val_dir and os.path.exists(val_dir):
            num_val_images = sum([len(files) for r, d, files in os.walk(val_dir) if any(f.endswith('.png') for f in files)])
        else:
            total_train_images = sum([len(files) for r, d, files in os.walk(train_dir) if any(f.endswith('.png') for f in files)])
            num_val_images = int(total_train_images * validation_split)  # 取整
            validation_steps = int(max(1, num_val_images // batch_size))
    else:
        validation_steps = None
    
    # 构建模型
    model, base_model = build_model(config)
    model.summary()
    
    # 第一阶段训练（冻结基模型）
    print("Stage 1: Training top layers with frozen backbone")
    base_model.trainable = False  # 确保冻结
    optimizer_stage1 = get_optimizer(config['training']['optimizer'], config['training']['learning_rate_stage1'])
    model.compile(optimizer=optimizer_stage1, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    callbacks_stage1 = [
        ModelCheckpoint(os.path.join(config['logging']['model_dir'], 'best_model_stage1.weights.h5'),
                        monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=True),
        EarlyStopping(monitor='val_accuracy', patience=config['training']['early_stop_patience'], restore_best_weights=True),
        CSVLogger(os.path.join(config['logging']['log_dir'], 'training_stage1.log'))
    ]
    
    # 学习率调度（如果是ReduceLROnPlateau则添加回调）
    if config['training']['lr_schedule_1'] == 'reduce_on_plateau':
        callbacks_stage1.append(ReduceLROnPlateau(monitor='val_loss', patience=config['training']['reduce_lr_patience'],
                                                   factor=config['training']['reduce_lr_factor'], verbose=1))
    
    history_stage1 = model.fit(
        train_dataset,
        epochs=config['training']['epochs_stage1'],
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=callbacks_stage1,
        verbose=1
    )
    
    # 第二阶段：微调（解冻基模型部分层）
    print("Stage 2: Fine-tuning with unfrozen backbone layers")
    finetune_layers = config['model']['finetune_layers']
    if finetune_layers == -1:
        base_model.trainable = True
    else:
        # 解冻最后 finetune_layers 层
        base_model.trainable = True
        for layer in base_model.layers[:-finetune_layers]:
            layer.trainable = False
    
    # 重新编译（使用更低学习率）
    optimizer_stage2 = get_optimizer(config['training']['optimizer'], config['training']['learning_rate_stage2'])
    model.compile(optimizer=optimizer_stage2, loss='sparse_categorical_crossentropy', metrics=['accuracy'])
    
    callbacks_stage2 = [
        ModelCheckpoint(os.path.join(config['logging']['model_dir'], 'best_model_final.weights.h5'),
                        monitor='val_accuracy', save_best_only=True, mode='max', save_weights_only=True),
        EarlyStopping(monitor='val_accuracy', patience=config['training']['early_stop_patience'], restore_best_weights=True),
        CSVLogger(os.path.join(config['logging']['log_dir'], 'training_stage2.log'))
    ]
    
    if config['training']['lr_schedule_2'] == 'reduce_on_plateau':
        callbacks_stage2.append(ReduceLROnPlateau(monitor='val_loss', patience=config['training']['reduce_lr_patience'],
                                                   factor=config['training']['reduce_lr_factor'], verbose=1))
    elif config['training']['lr_schedule_2'] == 'cosine':
        # 使用余弦退火调度，需要自定义回调或使用LearningRateScheduler
        # 简单使用CosineDecay作为优化器的学习率调度，但这样每步衰减，不是每epoch回调
        # 这里采用LearningRateScheduler模拟余弦衰减（每epoch调整）
        total_epochs = config['training']['epochs_stage2']
        initial_lr = config['training']['learning_rate_stage2']
        def cosine_scheduler(epoch, lr):
            progress = (epoch) / total_epochs
            cosine_decay = 0.5 * (1 + np.cos(np.pi * progress))
            return initial_lr * cosine_decay
        callbacks_stage2.append(LearningRateScheduler(cosine_scheduler, verbose=1))
    
    history_stage2 = model.fit(
        train_dataset,
        epochs=config['training']['epochs_stage2'],
        steps_per_epoch=steps_per_epoch,
        validation_data=val_dataset,
        validation_steps=validation_steps,
        class_weight=class_weights,
        callbacks=callbacks_stage2,
        verbose=1
    )
    
    # 重新构建模型（使用相同的 build_model 函数）
    model, _ = build_model(config)
    # 加载权重
    model.load_weights(os.path.join(config['logging']['model_dir'], 'best_model_final.weights.h5'))
    # 如果需要评估，则编译（可选）
    model.compile(optimizer=config['training']['optimizer'], loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    # 在验证集上评估
    y_true = []
    y_pred = []
    for images, labels in val_dataset:
        preds = model.predict(images)
        y_true.extend(labels.numpy())
        y_pred.extend(np.argmax(preds, axis=1))
        if len(y_true) >= num_val_images:
            break
    y_true = y_true[:num_val_images]
    y_pred = y_pred[:num_val_images]
    
    # 保存混淆矩阵和分类报告
    plot_confusion_matrix(y_true, y_pred, class_names,
                          os.path.join(config['logging']['output_dir'], 'confusion_matrix.png'))
    save_classification_report(y_true, y_pred, class_names,
                               os.path.join(config['logging']['output_dir'], 'classification_report.txt'))
    
    # 绘制训练曲线（合并两个阶段的history）
    combined_history = history_stage1.history
    for k, v in history_stage2.history.items():
        combined_history.setdefault(k, []).extend(v)
    plot_training_history(combined_history,
                          os.path.join(config['logging']['output_dir'], 'training_curves.png'))
    
    print("Training completed successfully.")

if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--config', type=str, default='config.yaml', help='Path to config file')
    args = parser.parse_args()
    main(args.config)