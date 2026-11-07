import json
from torch.utils.data import Dataset


class SchedulingDataset(Dataset):
    def __init__(self, file_path, tokenizer):
        self.tokenizer = tokenizer

        with open(file_path, "r") as f:
            self.data = [json.loads(line) for line in f]

    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        example = self.data[idx]

        text = example["text"]
        reference_date = example["reference_date"]
        label = json.dumps(example["label"])

        messages = [
            {
                "role": "system",
                "content": (
                    "Extract employee scheduling constraints as JSON. "
                    f"Today's date is {reference_date}."
                ),
            },
            {"role": "user", "content": text},
        ]

        prompt_ids = self.tokenizer.apply_chat_template(
            messages,
            tokenize=True,
            add_generation_prompt=True,
            return_dict=False,
        )

        full_ids = self.tokenizer.apply_chat_template(
            messages + [{"role": "assistant", "content": label}],
            tokenize=True,
            add_generation_prompt=False,
            return_dict=False,
        )

        labels = [-100] * len(prompt_ids) + full_ids[len(prompt_ids) :]
        assert full_ids[: len(prompt_ids)] == prompt_ids

        return {"input_ids": full_ids, "labels": labels}
