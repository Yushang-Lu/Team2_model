# TensorFlow 2.x 轻量级图像分类项目

这个项目基于 TensorFlow 2.x 和迁移学习，实现 `64x64` 彩色图像的 3 分类。默认使用 `MobileNetV3Small` 的 ImageNet 预训练权重，兼顾轻量、训练稳定和 TFLite 导出友好。

## 项目特性

- 使用 `MobileNetV3Small` 作为固定 backbone
- 支持目录式三分类数据集，兼容 `png/jpg/jpeg`
- 提供两阶段训练：冻结特征提取器 + 局部微调
- 自动保存最佳 `.keras` 模型、类别名称、训练日志和评估图表
- 提供单图预测、TFLite 导出、TFLite 测试和 Keras/TFLite 对比评估脚本

## 目录结构

```text
Team2_model/
├── config.yaml
├── train.py
├── predict.py
├── export_tflite.py
├── evaluate_tflite.py
├── compare_keras_tflite.py
├── models/
│   └── model_builder.py
├── utils/
│   ├── data_utils.py
│   ├── eval_utils.py
│   └── visualize.py
├── artifacts/          # 训练后生成
├── logs/               # 训练后生成
└── reports/            # 训练后生成
```

## 环境准备

建议使用 conda，并使用 Python 3.10。

```bash
conda create -n tf2-image-cls python=3.10 -y
conda activate tf2-image-cls
pip install -r requirements.txt
```

## 数据集组织

将数据按以下目录放置：

```text
data/
├── train/
│   ├── class0/
│   │   ├── sample_001.png
│   │   ├── sample_002.jpg
│   │   └── ...
│   ├── class1/
│   └── class2/
└── val/                # 可选
    ├── class0/
    ├── class1/
    └── class2/
└── test/               # 可选，供 TFLite 测试优先使用
    ├── class0/
    ├── class1/
    └── class2/
```

说明：

- 训练时必须恰好存在 3 个类别目录
- 如果没有 `data/val`，脚本会按固定随机种子 `42` 从训练集做“按类别分层随机切分”，默认抽取每类 `15%` 作为验证集
- `evaluate_tflite.py` 会优先使用 `data/test`，其次 `data/val`，最后回退到训练集中的验证切分
- 支持 `png`、`jpg`、`jpeg`

## 配置说明

默认配置位于 `config.yaml`：

- `data`：数据路径、输入尺寸、batch size、随机种子
- `model`：固定 backbone、预训练权重和 dropout
- `training`：两阶段 epoch、学习率、微调层数和回调参数
- `paths`：模型、日志和报告输出目录
- `export`：TFLite 默认导出配置

## 训练模型

```bash
python train.py --config config.yaml
```

训练脚本会自动完成：

- 加载目录式数据集并保存 `artifacts/class_names.json`
- 阶段 1：冻结 backbone 训练分类头
- 阶段 2：解冻最后 60 层做微调
- 选择验证集表现更好的阶段模型，保存为 `artifacts/best_model.keras`
- 输出训练曲线、混淆矩阵、分类报告和阶段日志

主要产物：

- `artifacts/best_model.keras`
- `artifacts/class_names.json`
- `artifacts/best_model_metrics.json`
- `logs/stage1_training.csv`
- `logs/stage2_training.csv`
- `reports/training_curves.png`
- `reports/confusion_matrix.png`
- `reports/classification_report.txt`

## 单图预测

```bash
python predict.py --model artifacts/best_model.keras --image path/to/image.png
```

如果 `class_names.json` 与模型放在同一目录下，预测脚本会自动读取；也可以手动指定：

```bash
python predict.py \
  --model artifacts/best_model.keras \
  --image path/to/image.jpg \
  --class-names artifacts/class_names.json
```

## 导出 TFLite

```bash
python export_tflite.py \
  --model artifacts/best_model.keras \
  --output artifacts/model.tflite \
  --quantize dynamic
```

说明：

- `dynamic`：启用动态量化，适合轻量部署
- `none`：不量化，保留浮点模型
- 导出脚本会同时复制 `class_names.json` 到输出目录，方便后续部署

## 测试 TFLite 模型并可视化

```bash
python evaluate_tflite.py \
  --model artifacts/model.tflite \
  --config config.yaml
```

默认输出目录是 `reports/tflite/`，会生成：

- `metrics.json`
- `classification_report.txt`
- `confusion_matrix.png`
- `per_class_accuracy.png`
- `confidence_histogram.png`

如果你有独立测试集，放到 `data/test/` 即可；脚本会优先使用它。

## 对比 .keras 和 .tflite 精度差异

```bash
python compare_keras_tflite.py \
  --keras-model artifacts/best_model.keras \
  --tflite-model artifacts/model.tflite \
  --config config.yaml
```

默认输出目录是 `reports/compare/`，会生成：

- `keras_metrics.json`
- `tflite_metrics.json`
- `comparison_summary.json`
- `keras_confusion_matrix.png`
- `tflite_confusion_matrix.png`
- `metric_comparison.png`
- `per_class_accuracy_comparison.png`
- `prediction_disagreements.csv`

其中 `comparison_summary.json` 会直接给出：

- `accuracy_gap`
- `macro_f1_gap`
- `weighted_f1_gap`
- `agreement_rate`
- `keras_only_correct_count`
- `tflite_only_correct_count`

## 默认训练策略

- 输入尺寸：`64x64x3`
- 类别数：`3`
- 验证切分：`15%`
- batch size：`32`
- 阶段 1：`12` 个 epoch，学习率 `1e-3`
- 阶段 2：`36` 个 epoch，学习率 `1e-4`
- 微调层数：backbone 最后 `12` 层
- dropout：`0.2`
- 数据增强：水平翻转、轻微旋转 `0.02`、轻微缩放 `0.05`
- 回调：`ModelCheckpoint`、`EarlyStopping`、`ReduceLROnPlateau`、`CSVLogger`

## 常见问题

1. 如果类别数不是 3，会怎样？

    脚本会直接报错，因为这个项目按三分类固定实现。

2. 如果图片存在损坏文件，会怎样？

    当前版本不会自动跳过坏图，会在读取时报错并提示具体路径，方便直接清理数据。
