import torch
import numpy as np
import warnings
from utils import load_data, get_features
from modules import DeTeCtiveClassifer
from transformers import AutoTokenizer
from sklearn.model_selection import train_test_split, StratifiedKFold, GridSearchCV
from sklearn.ensemble import RandomForestClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import SGDClassifier, LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.ensemble import StackingClassifier
from sklearn.metrics import classification_report
from sklearn.decomposition import PCA
from sklearn.neighbors import KNeighborsClassifier
from sklearn.feature_extraction.text import TfidfVectorizer

warnings.filterwarnings("ignore", category=UserWarning, module="xgboost")

# ================== 1. 加载 BERT 特征提取器 ==================
texts, _ = load_data("data/train.jsonl", lines=True)
tfidf_vectorizer = TfidfVectorizer(max_features=512)
tfidf_vectorizer.fit(texts)

tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
model = DeTeCtiveClassifer("bert-large-uncased", tfidf_vectorizer, 2).to("cuda:0")
model.load_state_dict(torch.load("DeTeCtive.pth", map_location="cuda:0"))
encoder = model.get_encoder()


# ================== 2. 定义数据加载函数 ==================
def get_train_data(ratio: float = 0.01):
    sub_texts, sub_labels = load_data("data/train.jsonl", lines=True, ratio=ratio)
    X_train = get_features(sub_texts, encoder, tokenizer, batch_size=32)
    torch.cuda.empty_cache()
    y_train = np.array(sub_labels)
    return X_train, y_train


# ================== 3. 定义模型调参函数 ==================
def tune_model(model, X, y, param_grid, name, n_jobs=1):
    grid_search = GridSearchCV(
        estimator=model,
        param_grid=param_grid,
        scoring="f1",
        cv=StratifiedKFold(5),
        n_jobs=n_jobs,
        verbose=3,
    )
    grid_search.fit(X, y)
    print(f"Best {name} parameters:", grid_search.best_params_)
    print(f"Best {name} F1 Score: {grid_search.best_score_:.4f}")
    return grid_search.best_estimator_


# ================== 4. 独立调参各模型 ==================
# 随机森林
rf_params = {
    "n_estimators": [100, 200],
    "max_depth": [None, 3, 5],
    "min_samples_split": [2, 5],
}
X, y = get_train_data(0.01)
rf_best = tune_model(
    RandomForestClassifier(random_state=42), X, y, rf_params, "rf", n_jobs=-1
)
torch.cuda.empty_cache()

# XGBoost
xgb_params = {
    "n_estimators": [100, 200],
    "max_depth": [5, 7, 10],
    "learning_rate": [0.1, 0.01],
    "subsample": [0.7, 0.8],
}
X, y = get_train_data(0.01)
xgb_best = tune_model(
    XGBClassifier(eval_metric="logloss", device="cuda"),
    X,
    y,
    xgb_params,
    "xgb",
    n_jobs=10,
)
torch.cuda.empty_cache()

# SVM (SGDClassifier)
svm_params = {
    "clf__alpha": [0.0001, 0.001, 0.01],
    "clf__max_iter": [500, 1000, 2000],
    "clf__penalty": ["l2", "elasticnet"],
}
X, y = get_train_data(0.01)
svm_best = tune_model(
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("clf", SGDClassifier(loss="hinge", penalty="l2", random_state=42)),
        ]
    ),
    X,
    y,
    svm_params,
    "svm",
    n_jobs=-1,
)
torch.cuda.empty_cache()

# KNN
knn_params = {
    "clf__n_neighbors": [3, 5, 7, 9],
    "clf__weights": [
        "uniform",
        "distance",
    ],
    "decomp__n_components": [0.8, 0.9, 0.95],
}
X, y = get_train_data(0.01)
knn_best = tune_model(
    Pipeline(
        [
            ("scaler", StandardScaler()),
            ("decomp", PCA()),
            ("clf", KNeighborsClassifier(algorithm="auto", metric="cosine")),
        ]
    ),
    X,
    y,
    knn_params,
    "knn",
    n_jobs=-1,
)
torch.cuda.empty_cache()

# ================== 5. 构建 StackingClassifier ==================
base_models = [("rf", rf_best), ("xgb", xgb_best), ("svm", svm_best), ("knn", knn_best)]

stacking_model = StackingClassifier(
    estimators=base_models, final_estimator=LogisticRegression()
)

# 对元模型进行简单调参
meta_params = {
    "final_estimator__C": [0.1, 1.0, 10.0],
    "final_estimator__penalty": ["l2"],
}

grid_search_stacking = GridSearchCV(
    estimator=stacking_model,
    param_grid=meta_params,
    scoring="f1",
    cv=StratifiedKFold(5),
    n_jobs=10,
    verbose=3,
)

X, y = get_train_data(0.01)
X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
grid_search_stacking.fit(X_train, y_train)
torch.cuda.empty_cache()

# ================== 6. 模型评估 ==================
y_pred = grid_search_stacking.predict(X_val)
print(classification_report(y_val, y_pred))
print("Best Stacking Parameters:", grid_search_stacking.best_params_)


# ================== 7. 生成提交文件 ==================
texts = load_data("data/test.jsonl", lines=True)
X_test = get_features(texts, encoder, tokenizer, batch_size=32)
torch.cuda.empty_cache()
preds = grid_search_stacking.predict(X_test)

with open("submit.txt", "w") as file:
    for pred in preds:
        file.write(f"{pred}\n")
