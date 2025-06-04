import torch
import pandas as pd


def load_data(data_path: str, lines: bool = False):
    data_frame = pd.read_json(data_path, lines=lines)
    texts = data_frame["text"].tolist()
    labels = data_frame["label"].tolist()
    return texts, labels


def get_features(encoder, tokenizer, texts: list, batch_size=8):
    all_cls_embeddings = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        )
        input_ids = inputs["input_ids"].to(encoder.device)
        attention_mask = inputs["attention_mask"].to(encoder.device)

        with torch.no_grad():
            features = encoder(input_ids=input_ids, attention_mask=attention_mask)

        cls_embeddings = features.last_hidden_state[:, 0, :]
        all_cls_embeddings.append(cls_embeddings.cpu())

    return torch.cat(all_cls_embeddings, dim=0).numpy()
