import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils import load_data, generate_database
from modules import DeTeCtiveClassifer, DeTeCtiveDataset, DeTeCtiveSampler

device = torch.device("cuda:0")

texts, labels = load_data("data/train.jsonl", lines=True, ratio=1.0)
tokenizer = AutoTokenizer.from_pretrained("bert-large-uncased")
model = DeTeCtiveClassifer("bert-large-uncased", num_classes=2).to(device)
dataset = DeTeCtiveDataset(texts, tokenizer, max_length=256, labels=labels)
sampler = DeTeCtiveSampler(dataset, 32)
dataloader = DataLoader(dataset, batch_sampler=sampler)

optimizer = optim.AdamW(model.parameters(), lr=3e-5)

for epoch in range(3):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch: {epoch + 1}"):
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

torch.save(model.state_dict(), "DeTeCtive.pth")