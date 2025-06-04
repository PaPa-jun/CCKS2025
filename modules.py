import torch, torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List


class SimCESModel(nn.Module):
    def __init__(self, encoder):
        super(SimCESModel, self).__init__()
        self.encoder = encoder

    def forward(self, input_ids, attention_mask=None):
        features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = features.last_hidden_state[:, 0, :]
        return cls_embedding


class InfoNCELoss(nn.Module):
    def __init__(self, temperature: float = 0.05):
        super(InfoNCELoss, self).__init__()
        self.temperature = temperature

    def forward(self, z1, z2):
        batch_size, hidden_size = z1.size()
        z = torch.cat([z1, z2], dim=0)
        z = F.normalize(z, p=2, dim=1)

        sim_matrix = torch.einsum("ik,jk->ij", z, z) / self.temperature

        # 构造正确标签
        indices = torch.arange(2 * batch_size, device=z.device)
        labels = torch.zeros_like(indices)

        # 前 batch_size 个样本的正样本索引
        forward_labels = torch.arange(batch_size, 2 * batch_size, device=z.device) - 1
        # 后 batch_size 个样本的正样本索引
        backward_labels = torch.arange(batch_size, device=z.device)

        labels[:batch_size] = forward_labels
        labels[batch_size:] = backward_labels

        # 构造 mask，排除对角线元素
        mask = ~torch.eye(2 * batch_size, dtype=torch.bool, device=z.device)
        logits = sim_matrix[mask].reshape(2 * batch_size, -1)

        loss = F.cross_entropy(logits, labels)
        return loss


class Classifier(nn.Module):
    def __init__(self, encoder, num_classes: int):
        super(Classifier, self).__init__()
        self.encoder = encoder
        self.classifier = nn.Sequential(
            nn.Linear(encoder.config.hidden_size, 512),
            nn.LayerNorm(512),
            nn.ReLU(),
            nn.Dropout(0.1),
            nn.Linear(512, num_classes),
        )

    def forward(self, input_ids, attention_mask=None):
        features = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        cls_embedding = features.last_hidden_state[:, 0, :]
        logits = self.classifier(cls_embedding)
        return logits


class TextDataset(Dataset):
    def __init__(
        self, texts: List[str], labels: List[str], tokenizer, max_length: int = 512
    ):
        super(TextDataset, self).__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        text = self.texts[index]
        label = self.labels[index]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
            "labels": torch.tensor(label, dtype=torch.long),
        }


class TestDataset(Dataset):
    def __init__(self, texts, tokenizer, max_length=512):
        self.texts = texts
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, idx):
        text = self.texts[idx]
        encoding = self.tokenizer(
            text,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }
