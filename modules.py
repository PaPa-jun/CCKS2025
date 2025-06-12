import torch, torch.nn as nn, string, nltk
import torch.nn.functional as F
from torch.utils.data import Dataset
from typing import List
from transformers import AutoModel
from sklearn.feature_extraction.text import TfidfVectorizer
from nltk.tokenize import word_tokenize, sent_tokenize

nltk.download("punkt", quiet=True)


class TextEmbeddingModel(nn.Module):
    def __init__(self, model_name: str, tfidf_vectorizer: TfidfVectorizer):
        super(TextEmbeddingModel, self).__init__()
        self.model = AutoModel.from_pretrained(model_name, trust_remote_code=True)
        self.tfidf_vectorizer = tfidf_vectorizer

    def _get_dl_features(self, input_ids: torch.Tensor, attention_mask: torch.Tensor):
        outputs = self.model(
            input_ids=input_ids,
            attention_mask=attention_mask,
            output_hidden_states=True,
        )
        cls_vector = outputs.last_hidden_state[:, 0, :]

        attention_mask_expanded = attention_mask.unsqueeze(-1).expand(
            outputs.last_hidden_state.size()
        )
        mean_pooling = torch.sum(
            outputs.last_hidden_state * attention_mask_expanded, dim=1
        ) / torch.clamp(attention_mask_expanded.sum(dim=1), min=1e-9)

        return torch.cat([cls_vector, mean_pooling], dim=-1)

    def _get_stochastic_features(self, texts):
        tfidf = torch.tensor(
            self.tfidf_vectorizer.transform(texts).toarray(),
            dtype=torch.float32,
            device=self.model.device,
        )

        features = []
        for text in texts:
            sentences = sent_tokenize(text)
            words = word_tokenize(text)
            unique_words = set(words)

            word_count = len(words)
            sentence_lengths = [len(sent.split()) for sent in sentences]
            avg_sentence_length_sample = (
                sum(sentence_lengths) / len(sentence_lengths) if sentences else 0
            )
            text_len = len(text)
            unique_word_count = len(unique_words)
            lexical_diversity = unique_word_count / word_count if word_count > 0 else 0
            punctuations = [char for char in text if char in string.punctuation]
            punctuation_type_count = len(set(punctuations))

            features.append(
                [
                    text_len,
                    word_count,
                    unique_word_count,
                    punctuation_type_count,
                    avg_sentence_length_sample,
                    lexical_diversity,
                ]
            )
        stocastics = F.normalize(
            torch.tensor(features, device=self.model.device), dim=-1
        )
        return torch.cat([tfidf, stocastics], dim=-1)

    def forward(
        self, input_ids: torch.Tensor, attention_mask: torch.Tensor, raw_texts: list
    ):
        dl_features = self._get_dl_features(input_ids, attention_mask)
        stocastic_features = self._get_stochastic_features(raw_texts)
        features = torch.cat([dl_features, stocastic_features], dim=-1)
        return features

    @property
    def hidden_size(self):
        return (
            self.model.config.hidden_size * 2
            + len(self.tfidf_vectorizer.get_feature_names_out())
            + 6
        )


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
        tfidf_vectorizer,
        num_classes: int,
        alpha: float = 0.5,
        beta: float = 0.5,
        temperature: float = 0.07,
    ):
        super(DeTeCtiveClassifer, self).__init__()
        self.encoder = TextEmbeddingModel(encoder_name, tfidf_vectorizer)
        self.classifier = ClassificationHead(
            input_dim=self.encoder.hidden_size,
            output_dim=num_classes,
            hidden_dim=self.encoder.hidden_size // 2,
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
                dim=-1,
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
        raw_texts: list,
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
        querys = self.encoder(
            input_ids=input_ids, attention_mask=attention_mask, raw_texts=raw_texts
        )
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
                "raw_texts": texts,
                "attention_mask": encoding["attention_mask"].flatten(),
                "labels": torch.tensor(labels, dtype=torch.long),
            }
        return {
            "input_ids": encoding["input_ids"].flatten(),
            "raw_texts": texts,
            "attention_mask": encoding["attention_mask"].flatten(),
        }
