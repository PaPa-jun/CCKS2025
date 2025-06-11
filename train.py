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
    accelerator="cuda", devices=3, strategy="ddp_find_unused_parameters_true"
)
fabric.launch()

tokenizer = AutoTokenizer.from_pretrained("/root/autodl-tmp/models/bert-base-uncased")
model = DeTeCtiveClassifer("/root/autodl-tmp/models/bert-base-uncased", num_classes=2)
optimizer = optim.AdamW(model.parameters(), lr=5e-6)
model, optimizer = fabric.setup(model, optimizer)

texts, labels = load_data("data/train.jsonl", lines=True, ratio=1.0)
dataset = DeTeCtiveDataset(texts, tokenizer, max_length=512, labels=labels)
dataloader = DataLoader(dataset, batch_size=64, shuffle=True)
dataloader = fabric.setup_dataloaders(dataloader)

for epoch in range(10):
    model.train()
    total_loss = 0.0
    for batch in tqdm(dataloader, desc=f"Epoch: {epoch + 1}", disable=(fabric.global_rank != 0)):
        inputs = {
            "input_ids": batch["input_ids"],
            "attention_mask": batch["attention_mask"],
            "labels": batch["labels"],
        }
        optimizer.zero_grad()
        loss, logits = model(**inputs)
        fabric.backward(loss)
        optimizer.step()
        total_loss += fabric.all_reduce(loss.item(), reduce_op="sum")

    fabric.print(f"Epoch {epoch + 1}, Loss: {total_loss / len(dataloader)}")
    torch.cuda.empty_cache()
    fabric.barrier()


if fabric.global_rank == 0:
    torch.save(model.state_dict(), "DeTeCtive_base_uncased.pth")
fabric.barrier()
