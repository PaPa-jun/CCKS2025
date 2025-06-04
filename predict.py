import torch, pandas as pd
from modules import Classifier, TestDataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader

model_path = "bert-base-uncased"
tokenizer_path = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = BertModel.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(model_path)
model = Classifier(encoder, num_classes=2).to(device)
model.load_state_dict(torch.load("model.pth", map_location=device))

texts = pd.read_json("test.jsonl", lines=True)["text"].tolist()

test_dataset = TestDataset(texts, tokenizer)
test_dataloader = DataLoader(test_dataset, batch_size=32, shuffle=False)
model.eval()
predictions = []
with torch.no_grad():
    for batch in test_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        logits = model(input_ids, attention_mask)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        predictions.extend(preds)


# Write predictions to file
with open("submit.txt", "w") as f:
    for pred in predictions:
        f.write(f"{pred}\n")
