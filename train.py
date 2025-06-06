import torch
import torch.optim as optim
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils import load_data, get_features
from modules import DeTeCtiveClassifer, DeTeCtiveDataset, DeTeCtiveSampler, Database

device = "cuda:0"

texts, labels = load_data("data/train.jsonl", lines=True)
tokenizer = AutoTokenizer.from_pretrained("../bert-base-uncased")
model = DeTeCtiveClassifer("../bert-base-uncased", num_classes=2).to(device)
dataset = DeTeCtiveDataset(texts, tokenizer, max_length=512, labels=labels)
sampler = DeTeCtiveSampler(dataset, 32)
dataloader = DataLoader(dataset, batch_sampler=sampler)

optimizer = optim.AdamW(model.parameters(), lr=3e-5)

for epoch in range(3):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        inputs = {
            "input_ids": batch["input_ids"].to(device),
            "attention_mask": batch["attention_mask"].to(device),
            "labels": batch["labels"].to(device),
        }
        optimizer.zero_grad()
        loss, logits = model(**inputs)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")


encoder = model.get_encoder()
ids, features = get_features(encoder, tokenizer, texts, 32, True)
db = Database(features.shape[1], True)

db.build_index(ids, features, labels)
db.save_db("database")
torch.save(model.state_dict(), "DeTeCtive.pth")