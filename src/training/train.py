import json
from functools import partial
from peft import LoraConfig, get_peft_model
import torch
from torch.utils.data import DataLoader
from utils import collate_fn
from transformers import AutoModelForCausalLM, AutoTokenizer
from dataset import SchedulingDataset

SEED = 42
torch.manual_seed(SEED)


def main():
    # CONFIG
    model_name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    BATCH_SIZE = 2

    train_dataset = SchedulingDataset("../../data/train.jsonl", tokenizer)
    train_loader = DataLoader(
        train_dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )

    val_dataset = SchedulingDataset("../../data/val.jsonl", tokenizer)
    val_loader = DataLoader(
        val_dataset,
        batch_size=BATCH_SIZE,
        shuffle=False,
        collate_fn=partial(collate_fn, pad_token_id=tokenizer.pad_token_id),
    )

    # Model Loading
    device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    model = model.to(device)
    print("Loading Model...")
    print("Device:", next(model.parameters()).device)
    print("Parameters:", model.num_parameters())

    # LoRA Config
    lora_config = LoraConfig(
        r=8,
        lora_alpha=16,
        lora_dropout=0.05,
        target_modules=["q_proj", "v_proj"],
        bias="none",
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, lora_config)
    print("\nLORA Trainable Parameters...")
    model.print_trainable_parameters()
    model.config.use_cache = False

    # training setup
    optimizer = torch.optim.AdamW(
        (p for p in model.parameters() if p.requires_grad), lr=2e-4
    )
    num_epochs = 3
    best_val_loss = float("inf")
    losses = {"train_loss": [], "val_loss": []}

    # training loop
    for epoch in range(num_epochs):
        # average train and val losses across batches
        avg_train_loss = train_epoch(model, train_loader, optimizer, device)
        avg_val_loss = validate(model, val_loader, device)

        # save checkpoint if val loss improves
        if avg_val_loss < best_val_loss:
            best_val_loss = avg_val_loss
            model.save_pretrained("checkpoints/best-lora")
            tokenizer.save_pretrained("checkpoints/best-lora")

        losses["train_loss"].append(avg_train_loss)
        losses["val_loss"].append(avg_val_loss)
        print(f"Epoch {epoch + 1}: Train={avg_train_loss:.4f}, Val={avg_val_loss:.4f}")

    with open("checkpoints/training_history.json", "w") as f:
        json.dump(losses, f, indent=2)


def train_epoch(model, train_loader, optimizer, device):
    model.train()
    train_loss = 0.0
    for i, batch in enumerate(train_loader):
        batch = {key: value.to(device) for key, value in batch.items()}
        optimizer.zero_grad()
        outputs = model(**batch)
        loss = outputs.loss
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
        optimizer.step()

        train_loss += loss.item()
        print(f"Batch {i + 1}/{len(train_loader)} complete")
        if (i + 1) % 10 == 0:
            print(
                f"Allocated: {torch.mps.current_allocated_memory() / 1024**3:.2f} GB | "
                f"Driver: {torch.mps.driver_allocated_memory() / 1024**3:.2f} GB"
            )
            print("\n")

    avg_train_loss = train_loss / len(train_loader)

    return avg_train_loss


def validate(model, val_loader, device):
    model.eval()
    val_loss = 0.0
    with torch.no_grad():
        for batch in val_loader:
            batch = {k: v.to(device) for k, v in batch.items()}
            outputs = model(**batch)
            val_loss += outputs.loss.item()

    avg_val_loss = val_loss / len(val_loader)
    return avg_val_loss


if __name__ == "__main__":
    main()
