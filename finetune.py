import torch, torch.nn as nn
import torch.optim as optim
from utils import load_data
from modules import Classifier, TextDataset
from transformers import BertTokenizer, BertModel
from torch.utils.data import DataLoader
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report

model_path = "bert-base-uncased"
tokenizer_path = "bert-base-uncased"
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

encoder = BertModel.from_pretrained(model_path)
tokenizer = BertTokenizer.from_pretrained(tokenizer_path)

texts, labels = load_data("data/train.jsonl", lines=True)
X_train, X_val, y_train, y_val = train_test_split(
    texts, labels, test_size=0.2, random_state=42
)

train_dataset = TextDataset(X_train, y_train, tokenizer)
val_dataset = TextDataset(X_val, y_val, tokenizer)
train_dataloader = DataLoader(train_dataset, batch_size=32, shuffle=True)
val_dataloader = DataLoader(val_dataset, batch_size=32, shuffle=False)

model = Classifier(encoder, num_classes=2).to(device)

# Freeze encoder's weights
for param in model.encoder.parameters():
    param.requires_grad = False

optimizer = optim.Adam(model.parameters(), lr=3e-6)
criterion = nn.CrossEntropyLoss()

for epoch in range(3):
    model.train()
    total_loss = 0.0
    for batch in train_dataloader:
        input_ids = batch["input_ids"].to(device)
        attention_mask = batch["attention_mask"].to(device)
        labels = batch["labels"].to(device)

        optimizer.zero_grad()
        logits = model(input_ids, attention_mask)
        loss = criterion(logits, labels)
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    print(f"Epoch {epoch + 1}, Training Loss: {total_loss / len(train_dataloader)}")

    model.eval()
    val_loss = 0.0
    val_preds = []
    val_labels = []
    with torch.no_grad():
        for batch in val_dataloader:
            input_ids = batch["input_ids"].to(device)
            attention_mask = batch["attention_mask"].to(device)
            labels = batch["labels"].to(device)

            logits = model(input_ids, attention_mask)
            loss = criterion(logits, labels)
            val_loss += loss.item()

            preds = logits.argmax(dim=-1)
            val_preds.extend(preds.cpu().numpy())
            val_labels.extend(labels.cpu().numpy())

    print(f"Epoch {epoch + 1}, Validation Loss: {val_loss / len(val_dataloader)}")
    print(
        f"Epoch {epoch + 1}, Validation Accuracy: {accuracy_score(val_labels, val_preds)}"
    )
    print(classification_report(val_labels, val_preds))

# Save the model
torch.save(model.state_dict(), "model.pth")
tokenizer.save_pretrained("finetuned_tokenizer")
