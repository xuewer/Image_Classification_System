# Image Classification System

一个完整的水果图像分类计算机视觉项目，覆盖数据整理、数据增强、迁移学习、模型训练、指标评估、错误分析和 Streamlit 在线 Demo。

## 1. 项目目标

本项目将图像分类任务实现为一条可复现 pipeline：

```text
原始图片
  -> 类别目录整理
  -> train / val / test 分层划分
  -> 数据审计
  -> 数据增强
  -> ResNet18/ResNet50 训练
  -> 验证集选最优 checkpoint
  -> 测试集 Accuracy / Precision / Recall / F1
  -> 混淆矩阵和错误样本分析
  -> Streamlit 图片上传预测
```

当前示例任务包含五类水果：

```text
apple
banana
grape
orange
strawberry
```

## 2. 目录结构

```text
image_classification_system/
├── app.py                    # Streamlit Web Demo
├── train.py                  # 训练、验证、checkpoint 和训练报告
├── evaluate.py               # 测试集指标和混淆矩阵
├── error_analysis.py         # 高置信度错误样本导出
├── audit_dataset.py          # 坏图、类别、跨 split 重复审计
├── pipeline.py               # 一键运行完整 pipeline
├── model.py                  # ResNet18 / ResNet50 构建
├── dataset.py                # ImageFolder 和数据增强
├── run_utils.py              # 路径和实验目录管理
├── requirements.txt
├── PROJECT_REPORT.md         # 详细项目分析与总结
├── data/
│   ├── raw/                  # 按类别存放原始图片
│   └── split/                # train / val / test
├── scripts/
│   └── split_dataset.py      # 数据集划分
└── outputs/
    └── test_XX/              # 每次训练独立实验目录
```

## 3. 环境安装

建议 Python 3.10 或 3.11。当前环境也支持 CPU 运行。

仓库仅包含源码、文档、测试和数据目录模板；本地图片、Parquet 数据集、训练权重和实验产物不随代码发布。首次克隆后，请先准备数据并训练模型，再使用评估脚本或 Web Demo。

```powershell
git clone https://github.com/xuewer/Image_Classification_System.git
cd Image_Classification_System
python -m venv .venv
.\.venv\Scripts\Activate.ps1
python -m pip install -r requirements.txt
```

检查 PyTorch：

```powershell
python -c "import torch, torchvision; print(torch.__version__); print(torchvision.__version__); print(torch.cuda.is_available())"
```

## 4. 数据准备

将图片放入类别目录：

```text
data/raw/
├── apple/
├── banana/
├── grape/
├── orange/
└── strawberry/
```

每个类别建议至少 200 张图片；真实项目建议 500 张以上，并覆盖：

- 不同光照
- 不同背景
- 不同距离和角度
- 不同大小
- 遮挡和部分目标
- 单个目标和多个目标
- 手机拍摄和网络图片

划分数据集：

```powershell
python scripts\split_dataset.py `
  --input data\raw `
  --output data\split `
  --train 0.7 `
  --val 0.15 `
  --test 0.15 `
  --seed 42
```

划分脚本保留类别内部的子目录，避免同名图片互相覆盖。输出目录只能为空或包含 `.gitkeep`；重新划分时请指定新目录，避免旧划分中的图片混入新集合。输入目录和输出目录不能重叠。

审计数据集：

```powershell
python audit_dataset.py --data data\split
```

审计报告默认保存到：

```text
data/split/audit_report.json
```

审计内容包括图片数量、类别数量、坏图和跨 train/val/test 的完全重复文件。注意：哈希审计不能发现视觉上相似但文件不同的图片，Fruits-360 等视频帧型数据仍需关注近似样本泄露。

缺失/空集合、空类别、类别不一致、坏图或跨集合完全重复图片会使审计以非零状态退出；`pipeline.py` 将停止执行，审计详情仍会写入报告。

## 5. 训练

### ImageNet 预训练迁移学习

```powershell
python train.py `
  --data data\split `
  --model resnet18 `
  --epochs 5 `
  --batch-size 16 `
  --lr 3e-4 `
  --min-lr 1e-6 `
  --device cpu
```

### 从零训练

```powershell
python train.py `
  --data data\split `
  --model resnet18 `
  --epochs 5 `
  --batch-size 16 `
  --lr 3e-4 `
  --min-lr 1e-6 `
  --device cpu `
  --no-pretrained
```

### ResNet50

```powershell
python train.py `
  --data data\split `
  --model resnet50 `
  --epochs 15 `
  --batch-size 8 `
  --device cpu
```

每次训练都会自动创建新的目录，例如：

```text
outputs/test_03/
├── best.pt
├── last.pt
├── classes.json
├── run_config.json
├── history.json
├── training_curves.png
└── analysis.txt
```

`best.pt` 按验证集准确率保存，`last.pt` 保存最后一轮模型。`analysis.txt` 会在训练结束时自动生成。

## 6. 一键运行

一键完成数据审计、训练、测试集评估和错误分析：

```powershell
python pipeline.py `
  --data data\split `
  --output outputs `
  --model resnet18 `
  --epochs 5 `
  --batch-size 16 `
  --device cpu
```

也可以运行从零训练：

```powershell
python pipeline.py `
  --data data\split `
  --epochs 5 `
  --batch-size 16 `
  --device cpu `
  --no-pretrained
```

## 7. 独立评估

评估指定模型：

```powershell
python evaluate.py `
  --data data\split\test `
  --checkpoint outputs\test_02\best.pt `
  --device cpu
```

默认输出到 checkpoint 所在目录：

```text
outputs/test_02/evaluation/
├── metrics.json
├── classification_report.json
├── classification_report.txt
└── confusion_matrix.png
```

输出指标：

- Accuracy
- Macro Precision
- Macro Recall
- Macro F1
- 每个类别的 Precision / Recall / F1 / Support
- 混淆矩阵

## 8. 错误分析

```powershell
python error_analysis.py `
  --data data\split\test `
  --checkpoint outputs\test_02\best.pt `
  --top-k 50 `
  --device cpu
```

错误样本默认保存到：

```text
outputs/test_02/errors/
├── summary.json
└── 001__true_class__pred_class__confidence__filename.jpg
```

错误分析拒绝覆盖非空输出目录；重复分析时可通过 `--output outputs/new_errors` 指定新目录。

错误分析适合回答：

- 哪些类别最容易混淆
- 错误是否集中在某个类别
- 模型是否存在高置信度误判
- 是否存在背景偏差、光照偏差或标签问题

## 9. Web Demo

```powershell
streamlit run app.py --server.port 8501
```

浏览器打开：

```text
http://localhost:8501
```

Demo 会自动加载 `outputs/` 下编号最大的可用实验目录中的 `best.pt`，支持：

- JPG / JPEG / PNG / WEBP 上传
- 原图预览
- 多视角推理
- Top-5 类别和置信度
- 当前 checkpoint 路径显示

如果新增了更好的实验目录，Demo 会自动切换到最新 checkpoint。

## 10. 当前实验结果

以下是本地已有实验 `outputs/test_01` 和 `outputs/test_02` 的历史结果；这些实验产物不随仓库发布：

| 实验    | 训练方式        | 最佳验证准确率 | 测试 Accuracy | 测试 Macro F1 |
| ------- | --------------- | -------------: | ------------: | ------------: |
| test_01 | 从零训练        |         99.46% |       100.00% |       100.00% |
| test_02 | ImageNet 预训练 |        100.00% |        99.73% |        99.73% |

`test_02` 的预训练模型收敛更快；但当前数据来自 Fruits-360 风格数据，随机文件切分可能包含同一拍摄序列的近似图片，因此还需要真实手机图片做外部测试。

详细分析见：

```text
PROJECT_REPORT.md
outputs/test_01/analysis.txt
outputs/test_02/analysis.txt
```

## 11. 项目总结

该项目完整实现了从数据到部署的计算机视觉 pipeline：

1. 使用类别目录组织图像数据。
2. 通过分层随机划分构造训练集、验证集和测试集。
3. 使用随机裁剪、水平翻转、旋转和颜色扰动进行数据增强。
4. 基于 PyTorch 和 ResNet 实现从零训练与迁移学习。
5. 使用验证集选择最佳 checkpoint。
6. 使用 Accuracy、Precision、Recall、F1 和混淆矩阵评估模型。
7. 导出错误样本并进行定性错误分析。
8. 通过 Streamlit 提供图片上传和在线预测。

真实项目落地前，应继续补充外部测试集、分组切分、更多真实场景数据和多随机种子实验。

## 12. 验证与提交范围

运行回归测试：

```powershell
python -m unittest discover -s tests -v
```

测试覆盖数据划分的可复现性、同名文件保留、覆盖保护、无效输入、坏图/重复图片/类别不一致审计，以及实验编号超过 99 和缺失 checkpoint 的处理。

`.gitignore` 排除数据图片、模型权重、训练结果、编辑器配置、虚拟环境、`.env`、私钥和 Streamlit secrets。目录中的 `.gitkeep` 和数据准备说明会保留。推送前仍应人工检查 `git diff --cached --name-only`，不要使用强制添加来提交私密文件。
