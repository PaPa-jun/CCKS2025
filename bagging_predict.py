import torch
import numpy as np
from utils import load_data, get_features
from modules import DeTeCtiveClassifer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.ensemble import BaggingClassifier
from sklearn.metrics import f1_score

# ================== 1. 加载 BERT 特征提取器 ==================
classifier = DeTeCtiveClassifer("/root/autodl-tmp/models/bert-base-uncased", 2).to("cuda:0")
classifier.load_state_dict(torch.load("DeTeCtive_base_uncased.pth", map_location="cuda:0"))
encoder = classifier.get_encoder()
tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/bert-base-uncased")

# ================== 2. 数据加载与特征提取 ==================
texts, labels = load_data("data/train.jsonl", lines=True)
X_train = get_features(texts, encoder, tokenizer, batch_size=32)
y_train = np.array(labels)

# 划分训练集和验证集（添加类别平衡）
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# ================== 3. 构建 KNN + Bagging 模型 ==================
knn_base = make_pipeline(
    StandardScaler(),
    PCA(n_components=0.95),
    KNeighborsClassifier(algorithm="auto", metric="cosine"),
)

# Bagging 集成模型（简化参数范围）
bagging_knn = BaggingClassifier(
    estimator=knn_base,
    random_state=42,
)

param_grid = {
    "estimator__kneighborsclassifier__n_neighbors": [3, 5, 7, 10],
    "estimator__kneighborsclassifier__weights": ["distance", "uniform"],
    "n_estimators": [10, 15],
    "max_samples": [0.6, 0.7, 0.8],
    "max_features": [0.6, 0.7, 0.8],
}

grid_search = GridSearchCV(
    estimator=bagging_knn,
    param_grid=param_grid,
    scoring="f1",
    cv=StratifiedKFold(n_splits=5, shuffle=True, random_state=42),
    n_jobs=-1,
    verbose=3,
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
texts_test = load_data("data/test.jsonl", lines=True)
X_test = get_features(texts_test, encoder, tokenizer, batch_size=32)

predictions = best_model.predict(X_test)
with open("submit_knn.txt", "w") as f:
    for pred in predictions:
        f.write(f"{pred}\n")
