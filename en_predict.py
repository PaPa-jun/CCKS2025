import torch
import xgboost as xgb
import pandas as pd
import numpy as np
from utils import load_data, get_features
from modules import DeTeCtiveClassifer
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel
from sklearn.ensemble import StackingClassifier, RandomForestClassifier
from sklearn.linear_model import LogisticRegression

# 加载 BERT 模型和 tokenizer（保持原样）
classifier = DeTeCtiveClassifer("../bert-base-uncased", 2).to("cuda")
classifier.load_state_dict(torch.load("DeTeCtive.pth", map_location="cuda"))
encoder = classifier.get_encoder()
# encoder = BertModel.from_pretrained("bert-base-uncased").to("cuda")
tokenizer = BertTokenizer.from_pretrained("../bert-base-uncased")

# 数据加载与特征提取（保持原样）
texts, labels = load_data("data/train.jsonl", lines=True)
_, X_train = get_features(encoder, tokenizer, texts, batch_size=32)
y_train = np.array(labels)
X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

# 定义基模型集合
estimators = [
    (
        "xgb",
        xgb.XGBClassifier(
            objective="binary:logistic",
            eval_metric="logloss",
            n_estimators=300,
            max_depth=3,
            learning_rate=0.1,
            subsample=0.8,
        ),
    ),
    ("rf", RandomForestClassifier(n_estimators=200, max_depth=5, random_state=42)),
    ("lr", LogisticRegression(penalty="l2", C=1.0, solver="liblinear")),
]

# 使用 Stackging 集成（元模型为逻辑回归）
model = StackingClassifier(
    estimators=estimators, final_estimator=LogisticRegression(C=2.0), cv=5, n_jobs=-1
)

# 训练模型
model.fit(X_train, y_train)

# 评估与预测（保持原样）
predictions = model.predict(X_val)
accuracy = np.mean(predictions == y_val)
print(f"Accuracy: {accuracy:.4f}")

# 测试集预测（保持原样）
text = pd.read_json("data/test.jsonl", lines=True)["text"].tolist()
X_test = get_features(encoder, tokenizer, text, batch_size=32)
predictions = model.predict(X_test)
with open("submit.txt", "w") as f:
    for pred in predictions:
        f.write(f"{pred}\n")
