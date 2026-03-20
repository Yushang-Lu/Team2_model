# 轻量化图像分类模型

本模型使用TensorFlow 2.x框架搭建，并使用迁移学习策略(MobileNetV3 Large)实现64×64彩色图像的3分类。

## 环境配置

推荐使用Miniconda管理Python环境。

```bash
# 创建并激活conda环境
conda create -n tf2 python=3.9
conda activate tf2

# 克隆项目并进入目录
git clone https://github.com/Yushang-Lu/Team2_model.git
cd Team2_model

# 安装依赖
pip install -r requirements.txt
```

## 数据集准备

将数据集按以下结构放置于data/目录下（否则需修改config.yaml中的路径）：

```txt
data/
├── train/
│   ├── class0/
│   │   ├── img1.png
│   │   └── ...
│   ├── class1/
│   └── class2/
└── val/               # 可选，若不存在则自动从训练集划分
    ├── class0/
    ├── class1/
    └── class2/
```

所有图像应为PNG格式，大小任意（代码会自动resize至64×64，但推荐64x64）。

## 配置文件

超参数在config.yaml中定义，可根据需要调整：

- data: 数据路径、增强策略、batch size等
- model: 基模型选择、冻结/微调策略
- training: 两阶段训练轮数、优化器、学习率调度等
- logging: 输出目录

## 训练模型

在项目根目录执行：

```bash
python train.py --config config.yaml
```

训练过程中会自动：

- 保存最佳模型至models/
- 记录日志至logs/
- 训练结束后生成混淆矩阵、分类报告及训练曲线至outputs/

## 模型导出为ONNX

训练完成后，使用以下命令将Keras模型转换为ONNX格式（支持动态batch维度）：

```bash
python export_onnx.py --weights models/best_model_final.weights.h5 --output models/model.onnx --config config.yaml
```

## 模型导出为TFLite

训练完成后，使用以下命令将Keras模型转换为TFLite格式：

```bash
python export_tflite.py --weights models/best_model_final.weights.h5 --output models/model.tflite --config config.yaml
```

## 模型部署

详见*逐飞科技*[推文](https://mp.weixin.qq.com/s/kESJdQ39PskYBtFpn8QhZw)
