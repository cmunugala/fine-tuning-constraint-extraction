import torch

BUCKETS = [384, 512, 640]


def collate_fn(batch, pad_token_id):
    max_length = max(len(item["input_ids"]) for item in batch)

    target_length = next((size for size in BUCKETS if size >= max_length), None)

    if target_length is None:
        raise ValueError(f"Sequence too long: {max_length}")

    input_ids = []
    labels = []
    attention_masks = []

    for item in batch:
        seq_length = len(item["input_ids"])
        padding_length = target_length - seq_length

        input_ids.append(item["input_ids"] + [pad_token_id] * padding_length)
        labels.append(item["labels"] + [-100] * padding_length)
        attention_masks.append([1] * seq_length + [0] * padding_length)

    return {
        "input_ids": torch.tensor(input_ids, dtype=torch.long),
        "labels": torch.tensor(labels, dtype=torch.long),
        "attention_mask": torch.tensor(attention_masks, dtype=torch.long),
    }
