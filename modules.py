import torch, torch.nn as nn
import torch.nn.functional as F
import numpy as np
from torch.utils.data import Dataset, Sampler
from typing import List
from transformers import AutoModel


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

        # 多层 [CLS] 拼接
        cls_embeddings = torch.cat(
            [hidden[:, 0, :] for hidden in outputs.hidden_states[-4:]], dim=-1
        )

        # 平均池化与最大池化
        last_hidden = outputs.last_hidden_state
        mask = attention_mask.unsqueeze(-1).expand(last_hidden.size()).float()
        mean_pool = torch.sum(last_hidden * mask, dim=1) / torch.clamp(
            mask.sum(dim=1), min=1e-9
        )
        max_pool, _ = torch.max(last_hidden * mask, dim=1)

        # 组合特征
        features = torch.cat((cls_embeddings, mean_pool, max_pool), dim=1)

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
    """
    Classifier based on DeTeCtive architecture for text classification tasks.
    """

    def __init__(
        self,
        encoder_name: str,
        num_classes: int,
        hidden_dim: int = 512,
        alpha: float = 0.5,
        beta: float = 0.5,
        temprature: float = 0.05,
    ):
        super(DeTeCtiveClassifer, self).__init__()
        self.encoder = TextEmbeddingModel(encoder_name)
        self.classifier = ClassificationHead(
            input_dim=4608,
            output_dim=num_classes,
            hidden_dim=hidden_dim,
        )
        self.alpha = nn.Parameter(torch.tensor(alpha))
        self.beta = nn.Parameter(torch.tensor(beta))
        self.temprature = nn.Parameter(torch.tensor(temprature))
        self.epsilon = torch.tensor(1e-8)

    def criterion(self, querys, keys, labels):
        querys = F.normalize(querys, dim=1)
        keys = F.normalize(keys, dim=1)
        cosine_similarity = torch.mm(querys, keys.t()) / self.temprature

        query_labels = labels.view(-1, 1)
        key_labels = labels.view(1, -1)

        positive_mask = (query_labels == key_labels).float()
        negative_mask = 1 - positive_mask

        positive_scores = torch.sum(
            cosine_similarity * positive_mask, dim=1
        ) / torch.max(positive_mask.sum(dim=1), self.epsilon)
        negative_scores = torch.sum(
            cosine_similarity * negative_mask, dim=1
        ) / torch.max(negative_mask.sum(dim=1), self.epsilon)

        constractive_logits = torch.stack([positive_scores, negative_scores], dim=1)
        constractive_labels = torch.zeros(
            constractive_logits.size(0),
            dtype=torch.long,
            device=constractive_logits.device,
        )
        constractive_loss = F.cross_entropy(constractive_logits, constractive_labels)

        classification_logits = self.classifier(querys)
        classification_loss = F.cross_entropy(classification_logits, labels)
        total_loss = self.alpha * constractive_loss + self.beta * classification_loss

        return total_loss, classification_logits

    def get_encoder(self):
        """
        Get the encoder part of the DeTeCtive classifier.
        :return: Encoder model.
        """
        return self.encoder.model

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
        keys = self.encoder(input_ids=input_ids, attention_mask=attention_mask)
        if labels is not None:
            loss, logits = self.criterion(querys, keys, labels)
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


class DeTeCtiveSampler(Sampler):
    def __init__(self, dataset: Dataset, batch_size: int = 32):
        super(DeTeCtiveSampler, self).__init__(dataset)
        self.dataset = dataset
        self.batch_size = batch_size
        self.class_indices = {0: [], 1: []}

        for idx, label in enumerate(dataset.labels):
            self.class_indices[
                label.item() if isinstance(label, torch.Tensor) else label
            ].append(idx)

        self.class_counts = {k: len(v) for k, v in self.class_indices.items()}
        self.min_class_size = min(self.class_counts.values())

    def __iter__(self):
        indices_class0 = np.random.permutation(self.class_indices[0])
        indices_class1 = np.random.permutation(self.class_indices[1])
        samples_per_class = self.batch_size // 2

        for i in range(len(self)):
            start0, end0 = i * samples_per_class, (i + 1) * samples_per_class
            start1, end1 = start0, end0
            batch_indices = []
            batch_indices.extend(indices_class0[start0:end0])
            batch_indices.extend(indices_class1[start1:end1])
            yield batch_indices

    def __len__(self):
        return self.min_class_size * 2 // self.batch_size
