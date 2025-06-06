import torch
import pandas as pd


def load_data(data_path: str, lines: bool = False):
    data_frame = pd.read_json(data_path, lines=lines)
    texts = data_frame["text"].tolist()
    if "label" in data_frame.columns:
        labels = data_frame["label"].tolist()
        return texts, labels
    return texts


def get_features(encoder, tokenizer, texts: list, batch_size: int = 32):
    all_features = []
    for i in range(0, len(texts), batch_size):
        batch_texts = texts[i : i + batch_size]
        inputs = tokenizer(
            batch_texts,
            truncation=True,
            padding="max_length",
            max_length=512,
            return_tensors="pt",
        ).to(encoder.device)
        input_ids = inputs["input_ids"]
        attention_mask = inputs["attention_mask"]

        with torch.no_grad():
            outputs = encoder(
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
        combined = torch.cat((cls_embeddings, mean_pool, max_pool), dim=1)
        all_features.append(combined.cpu())

    return torch.cat(all_features, dim=0).numpy()
