# Lightweight Image Classifier

一个基于 TensorFlow 的轻量化图像分类项目，专为 **64x64 彩色图像、3 分类** 任务设计。  
核心模型为 **10 层深度可分离卷积神经网络**，参数量极少，适合快速训练与边缘部署。  
项目包含完整训练流程、超参数配置、数据增强、训练可视化，并支持导出 **ONNX** 格式。

---

## 🚀 环境配置（Miniconda 用户）

本项目已在 Python 3.8 + TensorFlow 2.13 环境下测试。推荐使用 Miniconda 创建独立环境。

```bash
# 1. 创建并激活虚拟环境
conda create -n tf python=3.8
conda activate tf

# 2. 克隆项目并进入目录
git clone https://github.com/Yushang-Lu/lightweight-3class-classifier.git
cd lightweight-3class-classifier

# 3. 安装依赖
pip install -r requirements.txt
```

若需 GPU 加速，请安装 `tensorflow-gpu` 并确保 CUDA 环境正确。

---

## 📂 数据准备

请将您的图像数据按照以下结构存放：

```txt
data/
├── train/
│   ├── class0/      # 第1类图像（如：猫）
│   ├── class1/      # 第2类图像（如：狗）
│   └── class2/      # 第3类图像（如：鸟）
└── val/
    ├── class0/
    ├── class1/
    └── class2/
```

图像无需预先调整大小，代码会自动缩放至 `64x64`。支持 `.jpg`、`.png` 等常见格式。

---

## ⚙️ 超参数调整

所有训练参数均通过 `config.yaml` 配置：

- `data`: 训练/验证目录
- `model`: 输入尺寸、类别数
- `training`: 学习率、批次大小、轮次、早停耐心值
- `augmentation`: 数据增强策略（旋转、平移、缩放、翻转等）
- `onnx`: ONNX 导出设置

您可根据数据集规模自由调整。

---

## 🏋️ 训练模型

在项目根目录下执行：

```bash
python train.py --config config.yaml
```

训练过程中：

- 最佳模型（验证准确率最高）自动保存为 `outputs/best_model.h5`
- 训练曲线图保存为 `outputs/training_curves.png`
- 最终模型导出为 ONNX 格式 `outputs/model.onnx`

若需使用自定义配置文件，通过 `--config` 指定路径。

---

## 📊 训练可视化

训练结束后会自动弹出损失曲线和准确率曲线图，并保存至 `outputs/training_curves.png`。  
您可以通过该图直观判断过拟合情况与模型收敛程度。

---

## 💾 模型导出与部署

### 保存为 .h5（Keras）

训练完成后 `outputs/best_model.h5` 即为最佳权重模型，可直接用 `tf.keras.models.load_model` 加载。

### 转换为 .onnx

训练脚本已集成 `tf2onnx`，会自动将最佳模型转换为 ONNX 格式。  
您也可以手动执行转换：

```bash
python -m tf2onnx.convert --keras outputs/best_model.h5 --output outputs/model.onnx --opset 13
```

ONNX 模型可被 OpenVINO、ONNX Runtime、TensorRT 等框架加载，便于端侧推理。

---

## 📝 注意事项

1. **数据集规模**：深度可分离卷积参数量极少，若训练集很小，建议增强数据增强强度；若数据量较大，可适当增加滤波器数量。
2. **类别数**：若您的任务不是3类，请修改 `config.yaml` 中 `num_classes` 并调整数据目录结构。
3. **内存不足**：可降低 `batch_size`。
4. **自定义数据增强**：直接在 `config.yaml` 中修改 `augmentation` 字段即可。

---

## 🤝 贡献

欢迎提交 Issue 或 Pull Request 改进本项目。

---

## 📄 许可证

MIT License

---
