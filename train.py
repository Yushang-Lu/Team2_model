import os
import argparse
import tensorflow as tf
from tensorflow.keras.callbacks import ModelCheckpoint, EarlyStopping, ReduceLROnPlateau
import tf2onnx

from model import build_model
from utils import load_config, create_data_generators, plot_training_history

def main(config_path):
    # 1. 加载配置
    config = load_config(config_path)
    
    # 2. 创建输出目录
    os.makedirs('outputs', exist_ok=True)
    
    # 3. 数据生成器
    train_gen, val_gen = create_data_generators(config)
    print(f"训练样本数: {train_gen.samples}, 验证样本数: {val_gen.samples}")
    print(f"类别映射: {train_gen.class_indices}")
    
    # 4. 构建模型
    model = build_model(
        input_shape=tuple(config['model']['input_size']),
        num_classes=config['model']['num_classes']
    )
    model.summary()
    
    # 5. 编译模型
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=config['training']['learning_rate']),
        loss='categorical_crossentropy',
        metrics=['accuracy']
    )
    
    # 6. 回调函数
    callbacks = [
        ModelCheckpoint(
            'outputs/best_model.h5',
            monitor='val_accuracy',
            save_best_only=True,
            mode='max',
            verbose=1
        ),
        EarlyStopping(
            monitor='val_loss',
            patience=config['training']['early_stop_patience'],
            restore_best_weights=True,
            verbose=1
        ),
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.5,
            patience=5,
            min_lr=1e-7,
            verbose=1
        )
    ]
    
    # 7. 训练模型
    history = model.fit(
        train_gen,
        epochs=config['training']['epochs'],
        validation_data=val_gen,
        callbacks=callbacks,
        verbose=1
    )
    
    # 8. 可视化训练曲线
    plot_training_history(history, save_path='outputs/training_curves.png')
    
    # 9. 保存最终模型（可选）
    model.save('outputs/final_model.h5')
    
    # 10. 将最佳模型转换为 ONNX
    best_model = tf.keras.models.load_model('outputs/best_model.h5')
    
    # 指定输入签名（动态批量）
    spec = (tf.TensorSpec((None, *config['model']['input_size']), tf.float32, name="input"),)
    
    # 转换并保存
    onnx_path = config['onnx']['output_path']
    model_proto, _ = tf2onnx.convert.from_keras(
        best_model,
        input_signature=spec,
        opset=config['onnx']['opset'],
        output_path=onnx_path
    )
    print(f"ONNX 模型已保存至 {onnx_path}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='训练轻量化图像分类模型')
    parser.add_argument('--config', type=str, default='config.yaml', help='配置文件路径')
    args = parser.parse_args()
    main(args.config)