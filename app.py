from flask import Flask, render_template, request, send_from_directory
import numpy as np
import joblib
import os

app = Flask(__name__)

# 检查模型是否存在
MODEL_PATH = 'models/iris_knn_model.pkl'
SCALER_PATH = 'models/scaler.pkl'
LABEL_ENCODER_PATH = 'models/label_encoder.pkl'

if not all(os.path.exists(path) for path in [MODEL_PATH, SCALER_PATH, LABEL_ENCODER_PATH]):
    print("错误: 模型文件未找到！请先运行 train.py 训练模型")
    exit(1)

# 加载模型和预处理工具
model = joblib.load(MODEL_PATH)
scaler = joblib.load(SCALER_PATH)
le = joblib.load(LABEL_ENCODER_PATH)

# 默认特征值
DEFAULT_FEATURES = {
    'sepal_length': 5.1,
    'sepal_width': 3.5,
    'petal_length': 1.4,
    'petal_width': 0.2
}

@app.route('/static/<path:filename>')
def serve_static(filename):
    return send_from_directory('static', filename)

@app.route('/', methods=['GET', 'POST'])

def index():
    # 初始化特征值
    feature_values = DEFAULT_FEATURES.copy()
    result_available = False
    error = None
    species = None
    confidences = None
    
    if request.method == 'POST':
        try:
            # 获取表单数据
            features = [
                float(request.form['sepal_length']),
                float(request.form['sepal_width']),
                float(request.form['petal_length']),
                float(request.form['petal_width'])
            ]
            
            # 更新特征值用于回显
            feature_values = {
                'sepal_length': request.form['sepal_length'],
                'sepal_width': request.form['sepal_width'],
                'petal_length': request.form['petal_length'],
                'petal_width': request.form['petal_width']
            }
            
            # 预处理
            scaled_features = scaler.transform([features])
            
            # 预测
            prediction = model.predict(scaled_features)
            probabilities = model.predict_proba(scaled_features)[0]
            
            # 解码预测结果
            species = le.inverse_transform(prediction)[0]
            confidences = {
                le.classes_[i]: f"{prob*100:.1f}%" 
                for i, prob in enumerate(probabilities)
            }
            
            result_available = True
            
        except Exception as e:
            error = f"输入错误: {str(e)}"
    
    return render_template('index.html', 
                          feature_values=feature_values,
                          result_available=result_available,
                          error=error,
                          species=species,
                          confidences=confidences)

if __name__ == '__main__':
    app.run(host='0.0.0.0', port=5000, debug=True)
