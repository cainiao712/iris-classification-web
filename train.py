import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.datasets import load_iris
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.neighbors import KNeighborsClassifier
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
from sklearn.pipeline import Pipeline
import joblib
import os

# 设置可视化风格
sns.set(style="whitegrid")
plt.rcParams['font.size'] = 12
plt.rcParams['figure.figsize'] = (10, 6)

def load_and_explore_data():
    """加载数据并进行探索性分析"""
    print("步骤1: 加载和探索数据")
    iris = load_iris()
    df = pd.DataFrame(iris.data, columns=iris.feature_names)
    df['species'] = iris.target
    df['species'] = df['species'].map({0: 'setosa', 1: 'versicolor', 2: 'virginica'})
    
    # 数据概览
    print("\n数据概览:")
    print(df.head())
    print(f"\n数据集形状: {df.shape}")
    print(f"\n类别分布:\n{df['species'].value_counts()}")
    
    # 保存数据集
    df.to_csv('iris_dataset.csv', index=False)
    
    # 可视化数据分布
    visualize_data(df)
    return df, iris

def visualize_data(df):
    """执行数据可视化"""
    print("\n生成数据可视化图表...")
    
    # 确保static目录存在
    os.makedirs('static', exist_ok=True)
    
    # 特征分布直方图
    plt.figure()
    df.drop('species', axis=1).hist()
    plt.suptitle('特征分布直方图')
    plt.tight_layout()
    plt.savefig('static/feature_distributions.png')
    
    # 特征与目标的关系
    plt.figure()
    sns.pairplot(df, hue='species', markers=['o', 's', 'D'])
    plt.suptitle('特征与目标关系图', y=1.02)
    plt.savefig('static/feature_target_relationships.png')
    
    # 特征相关性热力图
    plt.figure()
    corr = df.drop('species', axis=1).corr()
    sns.heatmap(corr, annot=True, cmap='coolwarm', fmt=".2f")
    plt.title('特征相关性热力图')
    plt.savefig('static/feature_correlation.png')
    
    # 箱线图
    plt.figure()
    df_melted = df.melt(id_vars='species', var_name='features', value_name='value')
    sns.boxplot(x='features', y='value', hue='species', data=df_melted)
    plt.xticks(rotation=45)
    plt.title('特征箱线图')
    plt.tight_layout()
    plt.savefig('static/feature_boxplots.png')
    plt.close('all')

def preprocess_data(df):
    """数据预处理"""
    print("\n步骤2: 数据预处理")
    
    # 分割特征和目标
    X = df.drop('species', axis=1).values
    y = df['species'].values
    
    # 编码标签
    le = LabelEncoder()
    y = le.fit_transform(y)
    
    # 划分训练集和测试集
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.25, random_state=42, stratify=y
    )
    
    # 标准化特征
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)
    X_test_scaled = scaler.transform(X_test)
    
    # 确保models目录存在
    os.makedirs('models', exist_ok=True)
    
    # 保存标准化器
    joblib.dump(scaler, 'models/scaler.pkl')
    joblib.dump(le, 'models/label_encoder.pkl')
    
    return X_train_scaled, X_test_scaled, y_train, y_test, scaler, le

def train_and_optimize_model(X_train, y_train):
    """训练模型并进行超参数优化"""
    print("\n步骤3: 模型训练与优化")
    
    # 创建流水线
    pipeline = Pipeline([
        ('classifier', KNeighborsClassifier())
    ])
    
    # 设置参数网格
    param_grid = {
        'classifier__n_neighbors': list(range(1, 30)),
        'classifier__weights': ['uniform', 'distance'],
        'classifier__p': [1, 2]  # 1:曼哈顿距离, 2:欧氏距离
    }
    
    # 网格搜索优化
    grid_search = GridSearchCV(
        estimator=pipeline,
        param_grid=param_grid,
        cv=5,  # 减少交叉验证折数以加快速度
        scoring='accuracy',
        n_jobs=-1,
        verbose=1
    )
    
    print("正在进行网格搜索优化...")
    grid_search.fit(X_train, y_train)
    
    # 最佳参数和模型
    print(f"\n最佳参数: {grid_search.best_params_}")
    print(f"最佳交叉验证准确率: {grid_search.best_score_:.4f}")
    
    best_model = grid_search.best_estimator_
    
    # 保存最佳模型
    joblib.dump(best_model, 'models/iris_knn_model.pkl')
    
    # 绘制K值优化曲线
    plot_k_optimization(grid_search)
    
    return best_model

def plot_k_optimization(grid_search):
    """绘制K值优化曲线"""
    results = pd.DataFrame(grid_search.cv_results_)
    k_results = results[results['param_classifier__weights'] == 'uniform']
    
    plt.figure()
    for p_value in [1, 2]:
        subset = k_results[k_results['param_classifier__p'] == p_value]
        plt.plot(subset['param_classifier__n_neighbors'], 
                 subset['mean_test_score'], 
                 label=f"曼哈顿距离" if p_value == 1 else "欧氏距离")
    
    best_k = grid_search.best_params_['classifier__n_neighbors']
    plt.axvline(best_k, color='r', linestyle='--', label=f'最优K值 ({best_k})')
    plt.xlabel('K值')
    plt.ylabel('交叉验证准确率')
    plt.title('K值优化曲线')
    plt.legend()
    plt.grid(True)
    plt.savefig('static/k_optimization_curve.png')
    plt.close()

def evaluate_model(model, X_test, y_test, le):
    """评估模型性能"""
    print("\n步骤4: 模型评估")
    
    # 预测
    y_pred = model.predict(X_test)
    
    # 计算准确率
    accuracy = accuracy_score(y_test, y_pred)
    print(f"测试集准确率: {accuracy:.4f}")
    
    # 分类报告
    print("\n分类报告:")
    print(classification_report(y_test, y_pred, target_names=le.classes_))
    
    # 混淆矩阵
    conf_mat = confusion_matrix(y_test, y_pred)
    plt.figure()
    sns.heatmap(conf_mat, annot=True, fmt='d', cmap='Blues',
                xticklabels=le.classes_, 
                yticklabels=le.classes_)
    plt.xlabel('预测标签')
    plt.ylabel('真实标签')
    plt.title('混淆矩阵')
    plt.savefig('static/confusion_matrix.png')
    plt.close()
    
    return accuracy

if __name__ == "__main__":
    print("开始训练鸢尾花分类模型...")
    
    # 确保输出目录存在
    os.makedirs('models', exist_ok=True)
    os.makedirs('static', exist_ok=True)
    
    df, iris = load_and_explore_data()
    X_train, X_test, y_train, y_test, scaler, le = preprocess_data(df)
    best_model = train_and_optimize_model(X_train, y_train)
    accuracy = evaluate_model(best_model, X_test, y_test, le)
    
    print("\n模型训练完成！")
    print(f"最终模型测试准确率: {accuracy:.4f}")
    print("生成的可视化图表已保存在static目录")
    print("模型文件已保存在models目录")
    print("现在可以运行 'python app.py' 启动Web应用")