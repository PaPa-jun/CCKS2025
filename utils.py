import torch, random
import pandas as pd
import torch.nn.functional as F
from tqdm import tqdm
from modules import Database, TextEmbeddingModel
from collections import Counter, defaultdict


def load_data(data_path: str, lines: bool = False, ratio: float = 1.0):
    data_frame = pd.read_json(data_path, lines=lines)
    texts = data_frame["text"].tolist()
    if "label" in data_frame.columns:
        labels = data_frame["label"].tolist()
        sampled_texts, sampled_labels = uniform_sample(texts, labels, ratio)
        print(f"Train dataset loaded. Size: {len(sampled_texts)}")
        return sampled_texts, sampled_labels
    print(f"Test dataset loaded. Size: {len(texts)}")
    return texts


def get_features(
    texts: list,
    encoder: TextEmbeddingModel,
    tokenizer,
    batch_size: int = 32,
    normalize: bool = False,
):
    all_features = []
    for index in tqdm(range(0, len(texts), batch_size), desc="Get features: "):
        batch_texts = texts[index : index + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).to(encoder.model.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            features = encoder(input_ids, attention_mask)

        all_features.append(features.cpu())
    all_features = torch.cat(all_features, dim=0)
    if normalize is True:
        all_features = F.normalize(all_features, dim=1)
    ids = [id for id in range(all_features.shape[0])]
    return ids, all_features.numpy()


def generate_database(
    texts: list,
    labels: list,
    encoder: TextEmbeddingModel,
    tokenizer,
    save_path: str = "database",
    use_gpu: bool = False,
):
    ids, features = get_features(texts, encoder, tokenizer, 16, normalize=True)
    database = Database(features.shape[1], use_gpu)
    database.build_index(ids, features, labels)
    database.save_db(save_path)
    return database


def uniform_sample(texts: list, labels: list, ratio: float):
    label_to_indices = defaultdict(list)
    for idx, label in enumerate(labels):
        label_to_indices[label].append(idx)

    sampled_indices = []

    for label, indices in label_to_indices.items():
        sample_size = round(len(indices) * ratio)
        if sample_size > 0:
            sampled_indices.extend(random.sample(indices, sample_size))

    texts_sampled = [texts[i] for i in sampled_indices]
    labels_sampled = [labels[i] for i in sampled_indices]

    return texts_sampled, labels_sampled
