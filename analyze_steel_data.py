import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 设置中文字体支持
plt.rcParams['font.sans-serif'] = ['SimHei', 'Microsoft YaHei', 'SimSun', 'Arial Unicode MS']
plt.rcParams['axes.unicode_minus'] = False

# 设置图表清晰度
plt.rcParams['figure.dpi'] = 300
plt.rcParams['savefig.dpi'] = 300

def load_data(file_path):
    """加载钢铁疲劳寿命数据集"""
    print(f"正在加载钢铁疲劳寿命数据: {file_path}")
    try:
        # 加载CSV文件
        df = pd.read_csv(file_path)
        print(f"数据集加载成功，形状: {df.shape}")
        return df
    except Exception as e:
        print(f"加载数据集时出错: {str(e)}")
        return None

def analyze_data(df):
    """分析数据集的基本统计信息"""
    print("\n数据集基本信息:")
    print(df.info())
    
    print("\n数据集统计摘要:")
    print(df.describe())
    
    print("\n检查缺失值:")
    missing_values = df.isnull().sum()
    print(missing_values[missing_values > 0] if missing_values.sum() > 0 else "无缺失值")

def plot_correlation_matrix(df, output_dir='.'):
    """绘制相关性矩阵热图"""
    print("\n计算特征相关性...")
    
    # 计算相关性矩阵
    corr_matrix = df.corr()
    
    # 创建热图
    plt.figure(figsize=(16, 14))
    
    # 绘制热图
    sns.heatmap(corr_matrix, annot=True, fmt='.2f', cmap='RdBu_r',
                linewidths=0.5, vmin=-1, vmax=1, square=True, cbar_kws={"shrink": .8})
    
    plt.title('特征相关性矩阵', fontsize=16)
    plt.tight_layout()
    plt.savefig(f"{output_dir}/correlation_matrix.png")
    plt.close()
    
    # 返回与目标变量(Fatigue)的相关性
    target_corr = corr_matrix['Fatigue'].sort_values(ascending=False)
    print("\n与疲劳寿命的相关性 (降序排列):")
    print(target_corr)
    
    return corr_matrix

def plot_feature_distributions(df, output_dir='.'):
    """绘制特征分布直方图"""
    print("\n绘制特征分布...")
    
    # 选择数值型特征
    numeric_features = df.select_dtypes(include=['int64', 'float64']).columns.tolist()
    
    # 移除第一列(Sl. No.)，因为它只是索引
    if 'Sl. No.' in numeric_features:
        numeric_features.remove('Sl. No.')
    
    # 计算需要的行数和列数
    n_features = len(numeric_features)
    n_cols = 4  # 每行4个图
    n_rows = (n_features + n_cols - 1) // n_cols  # 向上取整
    
    # 创建子图
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(20, 5*n_rows))
    axes = axes.flatten()  # 将多维数组展平为一维
    
    # 为每个特征绘制分布图
    for i, feature in enumerate(numeric_features):
        if i < len(axes):  # 确保不超出子图数量
            sns.histplot(df[feature], kde=True, ax=axes[i])
            axes[i].set_title(feature)
            axes[i].set_xlabel('')
            axes[i].set_ylabel('Count')
    
    # 隐藏未使用的子图
    for j in range(i+1, len(axes)):
        axes[j].set_visible(False)
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/feature_distributions.png")
    plt.close()

def main():
    """主函数"""
    print("开始钢铁疲劳寿命数据分析...")
    
    # 设置数据路径
    data_path = "data/gt_data.csv"
    
    # 创建输出目录
    output_dir = "steel_fatigue_analysis"
    os.makedirs(output_dir, exist_ok=True)
    
    # 加载数据
    df = load_data(data_path)
    
    if df is None:
        print("数据加载失败，退出程序")
        return
    
    # 分析数据
    analyze_data(df)
    
    # 绘制相关性矩阵
    corr_matrix = plot_correlation_matrix(df, output_dir)
    
    # 绘制特征分布
    plot_feature_distributions(df, output_dir)
    
    print("\n钢铁疲劳寿命数据分析完成！")
    print(f"所有结果已保存至: {output_dir}")

if __name__ == "__main__":
    main()
