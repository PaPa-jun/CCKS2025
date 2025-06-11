import torch
import xgboost as xgb
import numpy as np
from utils import load_data, get_features
from modules import DeTeCtiveClassifer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier, StackingClassifier
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.metrics import f1_score
from sklearn.neighbors import KNeighborsClassifier

# ================== 1. 加载 BERT 特征提取器 ==================
classifier = DeTeCtiveClassifer("../models/bert-base-uncased", 2).to("cuda:0")
classifier.load_state_dict(
    torch.load("DeTeCtive_base_uncased.pth", map_location="cuda:0")
)
encoder = classifier.get_encoder()
tokenizer = AutoTokenizer.from_pretrained("../models/bert-base-uncased")

# ================== 2. 数据加载与特征提取 ==================
texts, labels = load_data("data/train.jsonl", lines=True)
X_train = get_features(texts, encoder, tokenizer, batch_size=32)
y_train = np.array(labels)

# 划分训练集和验证集
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42, stratify=y_train
)

# ================== 3. 统一交叉验证策略 ==================
cv_strategy = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)

# ================== 4. 分阶段独立调参 ==================
# ---- (1) XGBoost 调参（扩展关键参数）----
xgb_params = {
    "n_estimators": [200, 300],
    "max_depth": [5, 7],
    "learning_rate": [0.01, 0.05],
    "subsample": [0.8],
    "colsample_bytree": [0.7, 0.8],
    "gamma": [0, 0.1],
    "reg_alpha": [0.1, 1.0],
    "reg_lambda": [0.1, 1.0],
}
xgb_grid = GridSearchCV(
    estimator=xgb.XGBClassifier(eval_metric="logloss"),
    param_grid=xgb_params,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=3,
)
print("Start training XGBoost Model.")
xgb_grid.fit(
    X_train, y_train, eval_set=[(X_val, y_val)]
)
best_xgb = xgb_grid.best_estimator_

# ---- (2) 随机森林调参 ----
rf_params = {"n_estimators": [200, 300], "max_depth": [7, 10]}
rf_grid = GridSearchCV(
    estimator=RandomForestClassifier(random_state=42),
    param_grid=rf_params,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=3,
)
print("Start training RandomForest Model.")
rf_grid.fit(X_train, y_train)
best_rf = rf_grid.best_estimator_

# ---- (3) SVM 调参 ----
sgd_params = {
    "sgdclassifier__loss": ["hinge"],
    "sgdclassifier__penalty": ["l2", "elasticnet"],
    "sgdclassifier__alpha": [0.0001, 0.001],  # 扩展搜索范围
    "sgdclassifier__learning_rate": ["optimal"],
}
sgd_grid = GridSearchCV(
    estimator=make_pipeline(
        StandardScaler(), SGDClassifier(random_state=42, max_iter=1000, tol=1e-3)
    ),
    param_grid=sgd_params,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
    verbose=3,
)
print("Start training SGDClassifier Model.")
sgd_grid.fit(X_train, y_train)
best_sgd = sgd_grid.best_estimator_

# ---- (4) KNN 调参（新增 PCA 降维）----
knn_params = {
    "pca__n_components": [0.9, 0.95],
    "kneighborsclassifier__n_neighbors": [3, 5],
    "kneighborsclassifier__weights": ["distance", "uniform"],
    "kneighborsclassifier__metric": ["cosine"],
}
knn_grid = GridSearchCV(
    estimator=make_pipeline(
        StandardScaler(),
        PCA(),
        KNeighborsClassifier(algorithm="auto"),
    ),
    param_grid=knn_params,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
)
print("Start training KNN Model with PCA.")
knn_grid.fit(X_train, y_train)
best_knn = knn_grid.best_estimator_

# ================== 5. 构建集成模型 ==================
estimators = [("xgb", best_xgb), ("rf", best_rf), ("svm", best_sgd), ("knn", best_knn)]

final_params = {"C": [0.1, 1.0, 10]}
final_estimator = GridSearchCV(
    LogisticRegression(penalty="l2", max_iter=1000),
    final_params,
    scoring="f1",
    cv=cv_strategy,
    n_jobs=-1,
)

model = StackingClassifier(
    estimators=estimators,
    final_estimator=final_estimator,
    cv=cv_strategy,
    n_jobs=-1,
    stack_method="auto",
)

# ================== 6. 训练与评估 ==================
print("Start training Stacking Model.")
model.fit(X_train, y_train)

y_pred = model.predict(X_val)
print("验证集 F1 Score:", f1_score(y_val, y_pred))

# ================== 7. 测试集预测 ==================
texts_test = load_data("data/test.jsonl", lines=True)
X_test = get_features(texts_test, encoder, tokenizer, batch_size=32)

predictions = model.predict(X_test)
with open("submit.txt", "w") as f:
    for pred in predictions:
        f.write(f"{pred}\n")
