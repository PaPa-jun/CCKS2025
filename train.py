import torch
import torch.optim as optim
from tqdm import tqdm
from torch.utils.data import DataLoader
from transformers import AutoTokenizer
from utils import load_data
from modules import DeTeCtiveClassifer, DeTeCtiveDataset
from lightning import Fabric

torch.set_float32_matmul_precision("medium")
fabric = Fabric(
    accelerator="cuda",
    devices=[0, 1, 2],
    strategy="ddp_find_unused_parameters_true",
    precision="16-mixed",
)
fabric.launch()
fabric.barrier()

fabric.seed_everything(125)
fabric.barrier()

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/bert-large-uncased")
model = DeTeCtiveClassifer("/root/autodl-tmp/models/bert-large-uncased", num_classes=2)
optimizer = optim.AdamW(model.parameters(), lr=2e-5)
model, optimizer = fabric.setup(model, optimizer)
fabric.barrier()

texts, labels = load_data("data/train.jsonl", lines=True, ratio=0.1)
dataset = DeTeCtiveDataset(texts, tokenizer, max_length=512, labels=labels)
dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
dataloader = fabric.setup_dataloaders(dataloader)
fabric.barrier()

for epoch in range(10):
    model.train()
    total_loss = 0.0
    for batch in tqdm(
        dataloader, desc=f"Epoch: {epoch + 1}", disable=(fabric.global_rank != 0)
    ):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        optimizer.zero_grad()
        loss, logits = model(**inputs)
        fabric.backward(loss)
        optimizer.step()
        total_loss += loss

    all_loss = fabric.all_gather(total_loss)
    fabric.print(
        f"Epoch {epoch + 1}, Loss: {all_loss.sum() / (len(dataloader) * fabric.world_size)}"
    )
    torch.cuda.empty_cache()
    fabric.barrier()

if fabric.global_rank == 0:
    torch.save(model.state_dict(), "DeTeCtive_large_uncased.pth")
fabric.barrier()
