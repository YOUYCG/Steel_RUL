# 钢材疲劳强度预测

## 项目概述

本项目旨在通过机器学习方法预测钢材的疲劳强度。通过分析钢材的化学成分、热处理参数和其他物理特性，建立模型来预测钢材在10^7循环下的疲劳强度。

## 数据集

数据集包含以下特征：
- 钢材的化学成分（C, Si, Mn, P, S, Ni, Cr, Cu, Mo等）
- 热处理参数（正火温度、淬火温度、回火温度等）
- 物理特性（减少比率、夹杂物面积比例等）
- 目标变量：10^7循环下的疲劳强度

## 模型

项目中实现了两种模型：
1. **随机森林回归模型**：提供了较高的R²值（0.9847）
2. **人工神经网络模型**：提供了较低的RMSE值（29.32）

## 文件说明

- `main.py`：主要的Python脚本，包含数据加载、处理、模型训练和评估
- `main.ipynb`：Jupyter Notebook版本的代码
- `gt_data.csv`：数据集文件
- `forest_reg.joblib`：保存的随机森林模型
- `ann_model.pkl`：保存的人工神经网络模型
- `analyze_steel_data.py`和`analyze_steel_fatigue.py`：数据分析脚本

## 使用方法

1. 安装所需的依赖：
```
pip install pandas numpy matplotlib seaborn tensorflow scikit-learn joblib
```

2. 运行主脚本：
```
python main.py
```

## 结果

- 随机森林模型：
  - RMSE: 135.49
  - R²: 0.9847
  
- 人工神经网络模型：
  - RMSE: 29.32
  - R²: 0.9792
