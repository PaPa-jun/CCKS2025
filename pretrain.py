import torch, pandas as pd
import torch.optim as optim
from utils import load_data
from modules import SimCESModel, TestDataset, InfoNCELoss
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader

model_path = "bert-base-uncased"
tokenizer_path = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = BertModel.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

texts, labels = load_data("train.jsonl", lines=True)
texts = (
    texts[: int(0.1 * len(texts))]
    + pd.read_json("test.jsonl", lines=True)["text"].tolist()
)

model = SimCESModel(encoder).to(device)

optimizer = optim.Adam(model.parameters(), lr=3e-5)
criterion = InfoNCELoss(temperature=0.05)

dataset = TestDataset(texts, tokenizer)
dataloader = DataLoader(dataset, batch_size=32, shuffle=True)

for epoch in range(3):
    model.train()
    total_loss = 0.0
    for batch in dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)

        optimizer.zero_grad()
        z1 = model(input_ids, attention_mask)
        z2 = model(input_ids, attention_mask)
        loss = criterion(z1, z2)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")

# Save the model
encoder.save_pretrained("simces_encoder")
tokenizer.save_pretrained("simces_tokenizer")
