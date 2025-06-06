import torch
from utils import load_data, get_features, knn_predict
from transformers import AutoTokenizer
from modules import DeTeCtiveClassifer, Database

device = torch.device("cuda:0")

texts = load_data("data/test.jsonl", lines=True)
model = DeTeCtiveClassifer("../bert-base-uncased", 2).to(device)
model.load_state_dict(torch.load("DeTeCtive.pth", map_location=device))
encoder = model.get_encoder()
tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased")

ids, features = get_features(encoder, tokenizer, texts, normalize=True)
db = Database(features.shape[1], use_gpu=False)
db.load_db("database")

knn_results = db.knn_search(features, 10)

predictions = knn_predict(knn_results)

with open("submit.txt", "w") as file:
    for prediction in predictions:
        file.write(f"{prediction}\n")