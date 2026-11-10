import sys

sys.path.insert(0, "..")
from eval import load_test_data, load_model, generate_prediction
from pathlib import Path
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "checkpoints/best-lora"
TEST_PATH = "../../../data/val.jsonl"
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PROMPT_PATH = Path(__file__).parent.parent / "prompts" / "constraint_prompt_v1.0.txt"
SYSTEM_PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8").strip()

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
test_data = load_test_data(TEST_PATH)
model = load_model(fine_tuned=False)
pred = generate_prediction(model, tokenizer, test_data[1])

print(pred)
