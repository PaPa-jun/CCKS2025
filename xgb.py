import xgboost as xgb
import pandas as pd
import numpy as np
from utils import load_data, get_features
from sklearn.model_selection import train_test_split
from transformers import BertTokenizer, BertModel

# encoder = BertModel.from_pretrained("simces_encoder").to("cuda")
# tokenizer = BertTokenizer.from_pretrained("simces_tokenizer")

_, labels = load_data("train.jsonl", lines=True)
# texts, labels = load_data("train.jsonl", lines=True)
# X_train = get_features(encoder, tokenizer, texts, batch_size=32)
# np.save("X_ces.npy", X_train)
X_train = np.load("X_ces.npy")
y_train = np.array(labels)

X_train, X_val, y_train, y_val = train_test_split(
    X_train, y_train, test_size=0.2, random_state=42
)

model = xgb.XGBClassifier(
    objective="binary:logistic",
    eval_metric="logloss",
    n_estimators=300,
    max_depth=3,
    learning_rate=0.1,
)

model.fit(X_train, y_train)

predictions = model.predict(X_val)
accuracy = np.mean(predictions == y_val)
print(f"Accuracy: {accuracy:.4f}")


encoder = BertModel.from_pretrained("simces_encoder").to("cuda")
tokenizer = BertTokenizer.from_pretrained("simces_tokenizer")

text = pd.read_json("test.jsonl", lines=True)["text"].tolist()
X_test = get_features(encoder, tokenizer, text, batch_size=32)

predictions = model.predict(X_test)
with open("submit.txt", "w") as f:
    for pred in predictions:
        f.write(f"{pred}\n")
