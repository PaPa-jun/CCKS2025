import torch, torch.nn as nn
import torch.nn.functional as F
import random
from torch.utils.data import Dataset, Sampler
from typing import List
from transformers import AutoModel
from collections import defaultdict


class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name: str):
        super(TextEmbeddingModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)

    def forward(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )

        features = outputs.last_hidden_state[:, -1, :]
        return features


class ClassificationHead(nn.Module):
    """
    Classification head for text classification tasks.
    """

    def __init__(self, input_dim: int, output_dim: int, hidden_dim: int = 512):
        super(ClassificationHead, self).__init__()
        self.dense1 = nn.Linear(input_dim, hidden_dim)
        self.desne2 = nn.Linear(hidden_dim, hidden_dim // 4)
        self.projection_head = nn.Linear(hidden_dim // 4, output_dim)

        self.init_weights()

    def init_weights(self):
        """
        Initialize weights of the classification head.
        """
        nn.init.xavier_uniform_(self.dense1.weight)
        nn.init.xavier_uniform_(self.desne2.weight)
        nn.init.xavier_uniform_(self.projection_head.weight)

    def forward(self, x):
        """
        Forward pass through the classification head.
        :param x: Input tensor of shape (batch_size, input_dim)
        :return: Output tensor of shape (batch_size, output_dim)
        """
        x = F.tanh(self.dense1(x))
        x = F.tanh(self.desne2(x))
        x = self.projection_head(x)
        return x


class DeTeCtiveClassifer(nn.Module):
    def __init__(
        self,
        encoder_name: str,
        num_classes: int,
        hidden_dim: int = 512,
        alpha: float = 1.0,
        beta: float = 1.0,
        temperature: float = 0.07,
    ):
        super(DeTeCtiveClassifer, self).__init__()
        self.encoder = TextEmbeddingModel(encoder_name)
        self.classifier = ClassificationHead(
            input_dim=self.encoder.model.config.hidden_size,
            output_dim=num_classes,
            hidden_dim=hidden_dim,
        )
        self.alpha = nn.Parameter(torch.tensor(alpha), requires_grad=False)
        self.beta = nn.Parameter(torch.tensor(beta), requires_grad=False)
        self.temperature = nn.Parameter(torch.tensor(temperature), requires_grad=False)
        self.epsilon = torch.tensor(1e-6)

    def criterion(self, querys: torch.Tensor, labels: torch.Tensor):
        querys = F.normalize(querys, dim=-1)
        similarity_matrix = torch.mm(querys, querys.t()) / self.temperature

        positive_mask = (labels.view(-1, 1) == labels.view(1, -1)) * (
            1 - torch.eye(labels.shape[0], device=querys.device, dtype=torch.float32)
        )
        negative_mask = labels.view(-1, 1) != labels.view(1, -1)

        positive_score = positive_mask * similarity_matrix
        negative_score = negative_mask * similarity_matrix

        constructive_loss = F.cross_entropy(
            torch.cat(
                (
                    (
                        torch.sum(positive_score, dim=1)
                        / torch.max(torch.sum(positive_mask, dim=1), self.epsilon)
                    ).unsqueeze(1),
                    negative_score,
                ),
                dim=1,
            ),
            torch.zeros(labels.shape[0], dtype=torch.long, device=querys.device),
        )

        logits = self.classifier(querys)
        classification_loss = F.cross_entropy(logits, labels)

        return self.alpha * constructive_loss + self.beta * classification_loss, logits

    def get_encoder(self):
        """
        Get the encoder part of the DeTeCtive classifier.
        :return: Encoder model.
        """
        return self.encoder

    def forward(
        self,
        input_ids: torch.Tensor,
        attention_mask: torch.Tensor,
        labels: torch.Tensor = None,
    ):
        """
        Forward pass of the DeTeCtive classifier.
        :param input_ids: Input token IDs of shape (batch_size, sequence_length)
        :param attention_mask: Attention mask of shape (batch_size, sequence_length)
        :param labels: Labels for classification, optional
        :return: If labels are provided, returns loss and logits; otherwise, returns logits.
        """
        querys = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if labels is not None:
            loss, logits = self.criterion(querys, labels)
            return loss, logits
        logits = self.classifier(querys)
        return logits


class DeTeCtiveDataset(Dataset):
    def __init__(
        self,
        texts: List[str],
        tokenizer,
        max_length: int = 512,
        labels: List[int] = None,
    ):
        super(DeTeCtiveDataset, self).__init__()
        self.texts = texts
        self.labels = labels
        self.tokenizer = tokenizer
        self.max_length = max_length

    def __len__(self):
        return len(self.texts)

    def __getitem__(self, index):
        texts = self.texts[index]
        labels = self.labels[index] if self.labels else None
        encoding = self.tokenizer(
            texts,
            truncation=True,
            padding="max_length",
            max_length=self.max_length,
            return_tensors="pt",
        )
        if labels is not None:
            return {
                "input_ids": encoding["input_ids"].flatten(),
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "attention_mask": encoding["attention_mask"].flatten(),
        }


class BalancedClassSampler(Sampler):
    def __init__(self, dataset, shuffle=True):
        self.dataset = dataset
        self.class_indices = defaultdict(list)
        for idx, label in enumerate(dataset.labels):
            self.class_indices[label].append(idx)
        
        self.classes = list(self.class_indices.keys())
        self.num_classes = len(self.classes)
        self.shuffle = shuffle
        self.min_samples = min(len(indices) for indices in self.class_indices.values())

    def __iter__(self):
        # 为每个类生成打乱后的索引列表
        indices = [list(self.class_indices[cls]) for cls in self.classes]
        if self.shuffle:
            for idx_list in indices:
                random.shuffle(idx_list)
        
        # 按样本位置遍历，确保每个类取相同数量的样本
        for i in range(self.min_samples):
            batch = []
            for cls_idx in range(self.num_classes):
                batch.append(indices[cls_idx][i])
            yield batch

    def __len__(self):
        return self.min_samples * self.num_classes
