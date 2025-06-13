import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader, DistributedSampler
from transformers import AutoTokenizer
from utils import load_data
from modules import DeTeCtiveClassifer, DeTeCtiveDataset
from lightning import Fabric
from sklearn.feature_extraction.text import TfidfVectorizer

torch.set_float32_matmul_precision("medium")
fabric = Fabric(
    accelerator="cuda",
    devices=[0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11],
    strategy="ddp_find_unused_parameters_true",
    precision="16-mixed",
)
fabric.launch()
fabric.barrier()

fabric.seed_everything(125)
fabric.barrier()

texts, labels = load_data("data/train.jsonl", lines=True, ratio=1.0)
tfidf_vectorizer = TfidfVectorizer(max_features=512)
tfidf_vectorizer.fit(texts)
fabric.barrier()

tokenizer = AutoTokenizer.from_pretrained("/bert-large-uncased")
model = DeTeCtiveClassifer("/bert-large-uncased", tfidf_vectorizer, num_classes=2)
optimizer = optim.AdamW(model.parameters(), lr=5e-5)
model, optimizer = fabric.setup(model, optimizer)
fabric.barrier()

dataset = DeTeCtiveDataset(texts, tokenizer, max_length=512, labels=labels)
sampler = DistributedSampler(dataset, fabric.world_size, fabric.global_rank)
dataloader = DataLoader(dataset, batch_size=16, sampler=sampler)
dataloader = fabric.setup_dataloaders(dataloader)
fabric.barrier()

for epoch in range(5):
    model.train()
    total_loss = 0.0
    for batch in tqdm(
        dataloader, desc=f"Epoch: {epoch + 1}", disable=(fabric.global_rank != 0)
    ):
        inputs = {
            "input_ids": batch["input_ids"],
            "raw_texts": batch["raw_texts"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        optimizer.zero_grad()
        loss, logits = model(**inputs)
        fabric.backward(loss)
        optimizer.step()
        total_loss += loss.item()

    all_loss = fabric.all_gather(total_loss)
    fabric.print(
        f"Epoch {epoch + 1}, Loss: {all_loss.sum() / (len(dataloader) * fabric.world_size):.4f}"
    )
    torch.cuda.empty_cache()
    fabric.barrier()

if fabric.global_rank == 0:
    torch.save(model.state_dict(), "DeTeCtive.pth")
fabric.barrier()
