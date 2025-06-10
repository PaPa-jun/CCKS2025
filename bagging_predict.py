import torch
import xgboost as xgb
import numpy as np
from utils import load_data, get_features
from modules import DeTeCtiveClassifer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score

# ================== 1. 加载 BERT 特征提取器 ==================
# 加载预训练模型和 tokenizer
classifier = DeTeCtiveClassifer("bert-large-uncased", 2).to("cuda:1")
classifier.load_state_dict(torch.load("DeTeCtive.pth", map_location="cuda:1"))
encoder = classifier.get_encoder()
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")

# ================== 2. 数据加载与特征提取 ==================
# 加载训练数据并提取 BERT 特征
texts, labels = load_data("data/train.jsonl", lines=True)
_, X_train = get_features(texts, encoder, tokenizer, batch_size=32)
y_train = np.array(labels)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# ================== 3. 构建 KNN + Bagging 模型 ==================
# 1. 创建 KNN 分类器
knn_base = make_pipeline(
    StandardScaler(),
    KNeighborsClassifier(metric="cosine", algorithm="brute")
)

# 2. 构建 Bagging 集成模型
bagging_knn = BaggingClassifier(
    estimator=knn_base,
    n_jobs=1,
    random_state=42
)

param_grid = {
    'estimator__kneighborsclassifier__n_neighbors': [3, 5, 7],
    'estimator__kneighborsclassifier__weights': ['uniform', 'distance'],
    'n_estimators': [5, 7, 9, 15],
    'max_samples': [0.7, 0.8, 0.9],
    'max_features': [0.7, 0.8, 0.9],
}

grid_search = GridSearchCV(
    estimator=bagging_knn,
    param_grid=param_grid,
    scoring='f1',
    cv=5,
    n_jobs=-1,
    verbose=3
)

# ================== 4. 训练与评估 ==================
print("Start training Bagging KNN Model.")
grid_search.fit(X_train, y_train)

best_params = grid_search.best_params_
best_model = grid_search.best_estimator_

# 验证集评估
y_pred = best_model.predict(X_val)
print("验证集 F1 Score:", f1_score(y_val, y_pred))

# ================== 5. 测试集预测 ==================
# 加载测试集并提取特征
texts_test = load_data("data/test.jsonl", lines=True)
_, X_test = get_features(texts_test, encoder, tokenizer, batch_size=32)

# 生成预测并保存结果
predictions = best_model.predict(X_test)
with open("submit_bagging_knn.txt", "w") as f:
    for pred in predictions:
        f.write(f"{pred}\n")
