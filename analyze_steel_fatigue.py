import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import mean_squared_error, mean_absolute_error, r2_score
import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import Dense, Dropout, Input, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

# 设置随机种子以确保可重复性
SEED = 42
np.random.seed(SEED)
tf.random.set_seed(SEED)

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
    mask = np.triu(np.ones_like(corr_matrix, dtype=bool))  # 创建上三角掩码
    
    # 绘制热图
    sns.heatmap(corr_matrix, mask=mask, annot=True, fmt='.2f', cmap='RdBu_r',
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

def build_mstfnet_model(input_dim, output_dim=1):
    """
    构建用于钢铁疲劳寿命预测的MSTFNet模型
    
    参数:
    - input_dim: 输入特征维度
    - output_dim: 输出维度 (默认为1，预测疲劳寿命)
    
    返回:
    - 构建好的模型
    """
    print("\n构建MSTFNet模型...")
    print(f"输入维度: {input_dim}, 输出维度: {output_dim}")
    
    # 输入层
    inputs = Input(shape=(input_dim,))
    
    # 第一层特征提取
    x = Dense(128, activation='relu')(inputs)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 第二层特征提取
    x = Dense(64, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.2)(x)
    
    # 第三层特征提取
    x = Dense(32, activation='relu')(x)
    x = BatchNormalization()(x)
    x = Dropout(0.1)(x)
    
    # 输出层 - 预测疲劳寿命
    outputs = Dense(output_dim, activation='linear')(x)
    
    # 创建模型
    model = Model(inputs=inputs, outputs=outputs, name="Steel_Fatigue_MSTFNet")
    print("模型构建完成.")
    
    return model

def train_model(X_train, y_train, X_val, y_val, output_dir='.'):
    """训练模型并评估"""
    print("\n准备训练模型...")
    
    # 构建模型
    model = build_mstfnet_model(X_train.shape[1])
    
    # 编译模型
    model.compile(
        optimizer=Adam(learning_rate=0.001),
        loss='mse',
        metrics=['mae']
    )
    
    # 定义回调函数
    callbacks = [
        # 早停
        EarlyStopping(
            monitor='val_loss',
            patience=30,
            restore_best_weights=True,
            verbose=1
        ),
        # 学习率降低
        ReduceLROnPlateau(
            monitor='val_loss',
            factor=0.2,
            patience=10,
            min_lr=0.000001,
            verbose=1
        ),
        # 模型检查点
        ModelCheckpoint(
            filepath=f"{output_dir}/best_model.h5",
            monitor='val_loss',
            save_best_only=True,
            verbose=1
        )
    ]
    
    # 训练模型
    history = model.fit(
        X_train, 
        y_train,
        validation_data=(X_val, y_val),
        epochs=200,
        batch_size=16,
        callbacks=callbacks,
        verbose=1
    )
    
    # 绘制训练历史
    plt.figure(figsize=(12, 5))
    
    # 损失曲线
    plt.subplot(1, 2, 1)
    plt.plot(history.history['loss'], label='训练损失')
    plt.plot(history.history['val_loss'], label='验证损失')
    plt.title('模型损失')
    plt.xlabel('Epoch')
    plt.ylabel('损失')
    plt.legend()
    
    # MAE曲线
    plt.subplot(1, 2, 2)
    plt.plot(history.history['mae'], label='训练MAE')
    plt.plot(history.history['val_mae'], label='验证MAE')
    plt.title('平均绝对误差')
    plt.xlabel('Epoch')
    plt.ylabel('MAE')
    plt.legend()
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/training_history.png")
    plt.close()
    
    return model, history

def evaluate_model(model, X_test, y_test, output_dir='.'):
    """评估模型性能"""
    print("\n评估模型性能...")
    
    # 预测测试集
    y_pred = model.predict(X_test)
    
    # 计算评估指标
    mse = mean_squared_error(y_test, y_pred)
    rmse = np.sqrt(mse)
    mae = mean_absolute_error(y_test, y_pred)
    r2 = r2_score(y_test, y_pred)
    
    print(f"均方误差 (MSE): {mse:.4f}")
    print(f"均方根误差 (RMSE): {rmse:.4f}")
    print(f"平均绝对误差 (MAE): {mae:.4f}")
    print(f"决定系数 (R²): {r2:.4f}")
    
    # 绘制预测vs真实值散点图
    plt.figure(figsize=(10, 6))
    plt.scatter(y_test, y_pred, alpha=0.5)
    
    # 添加对角线
    min_val = min(y_test.min(), y_pred.min())
    max_val = max(y_test.max(), y_pred.max())
    plt.plot([min_val, max_val], [min_val, max_val], 'r--')
    
    plt.title('预测值 vs 真实值')
    plt.xlabel('真实疲劳寿命')
    plt.ylabel('预测疲劳寿命')
    plt.grid(True, linestyle='--', alpha=0.7)
    
    # 添加评估指标文本
    plt.text(0.05, 0.95, f'RMSE: {rmse:.2f}\nMAE: {mae:.2f}\nR²: {r2:.4f}',
             transform=plt.gca().transAxes, fontsize=12,
             verticalalignment='top', bbox=dict(boxstyle='round', alpha=0.1))
    
    plt.tight_layout()
    plt.savefig(f"{output_dir}/prediction_results.png")
    plt.close()
    
    # 返回评估指标
    return {
        'mse': mse,
        'rmse': rmse,
        'mae': mae,
        'r2': r2
    }

def main():
    """主函数"""
    print("开始钢铁疲劳寿命数据分析与预测...")
    
    # 设置数据路径
    data_path = "data/gt_data.csv"
    
    # 创建输出目录
    import os
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
    
    # 准备训练数据
    print("\n准备训练数据...")
    
    # 移除第一列(Sl. No.)，因为它只是索引
    if 'Sl. No.' in df.columns:
        df = df.drop('Sl. No.', axis=1)
    
    # 分离特征和目标变量
    X = df.drop('Fatigue', axis=1)
    y = df['Fatigue']
    
    # 标准化特征
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)
    
    # 分割数据集
    X_train, X_temp, y_train, y_temp = train_test_split(X_scaled, y, test_size=0.3, random_state=SEED)
    X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=SEED)
    
    print(f"训练集: {X_train.shape[0]} 样本")
    print(f"验证集: {X_val.shape[0]} 样本")
    print(f"测试集: {X_test.shape[0]} 样本")
    
    # 训练模型
    model, history = train_model(X_train, y_train, X_val, y_val, output_dir)
    
    # 评估模型
    metrics = evaluate_model(model, X_test, y_test, output_dir)
    
    print("\n钢铁疲劳寿命数据分析与预测完成！")
    print(f"所有结果已保存至: {output_dir}")

if __name__ == "__main__":
    main()
