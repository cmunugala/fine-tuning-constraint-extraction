import gc
import json
from collections import Counter
from pathlib import Path

import torch
from peft import PeftModel
from transformers import AutoModelForCausalLM, AutoTokenizer

# Config
MODEL_NAME = "Qwen/Qwen2.5-0.5B-Instruct"
ADAPTER_PATH = "checkpoints/best-lora"
TEST_PATH = "../../data/val.jsonl"
OUTPUT_DIR = Path("experiments")
RUN_NAME = "baseline"
MAX_NEW_TOKENS = 384
DEVICE = torch.device("mps" if torch.backends.mps.is_available() else "cpu")
PROMPT_PATH = Path(__file__).parent / "prompts" / "constraint_prompt_v1.0.txt"
SYSTEM_PROMPT_TEMPLATE = PROMPT_PATH.read_text(encoding="utf-8").strip()


def load_test_data(path):
    with open(path) as f:
        return [json.loads(line) for line in f]


def load_model(fine_tuned=False):
    model = AutoModelForCausalLM.from_pretrained(
        MODEL_NAME,
        dtype=torch.float32,
    )

    if fine_tuned:
        model = PeftModel.from_pretrained(model, ADAPTER_PATH)

    model = model.to(DEVICE)
    model.eval()

    return model


def parse_json(text):
    try:
        text = text.strip()
        if text.startswith("```"):
            text = text.split("\n", 1)[1]
            text = text.rsplit("```", 1)[0].strip()
        return json.loads(text)

    except (json.JSONDecodeError, IndexError, AttributeError):
        return None


def generate_prediction(model, tokenizer, example):
    messages = [
        {
            "role": "system",
            "content": SYSTEM_PROMPT_TEMPLATE.format(
                reference_date=example["reference_date"]
            ),
        },
        {
            "role": "user",
            "content": example["text"],
        },
    ]

    inputs = tokenizer.apply_chat_template(
        messages,
        tokenize=True,
        add_generation_prompt=True,
        return_dict=True,
        return_tensors="pt",
    )

    inputs = {k: v.to(DEVICE) for k, v in inputs.items()}

    with torch.inference_mode():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            do_sample=False,
            pad_token_id=tokenizer.pad_token_id,
        )

    # Decode only newly generated tokens
    generated_ids = output_ids[0, inputs["input_ids"].shape[1] :]

    return tokenizer.decode(
        generated_ids,
        skip_special_tokens=True,
    ).strip()


def parse_prediction(text):
    prediction = parse_json(text)

    if not isinstance(prediction, dict):
        return None

    constraints = prediction.get("constraints")

    if not isinstance(constraints, list):
        return None

    if not all(isinstance(c, dict) for c in constraints):
        return None

    return prediction


def normalize(value):
    """Ignore object key order and optional null fields."""
    if isinstance(value, dict):
        return {k: normalize(v) for k, v in sorted(value.items()) if v is not None}

    if isinstance(value, list):
        return [normalize(v) for v in value]

    return value


def constraint_counts(predicted, expected):
    """Count exact matches of individual constraints."""

    pred_constraints = [
        json.dumps(normalize(c), sort_keys=True) for c in predicted["constraints"]
    ]

    true_constraints = [
        json.dumps(normalize(c), sort_keys=True) for c in expected["constraints"]
    ]

    pred_counter = Counter(pred_constraints)
    true_counter = Counter(true_constraints)

    tp = sum((pred_counter & true_counter).values())
    fp = sum((pred_counter - true_counter).values())
    fn = sum((true_counter - pred_counter).values())

    exact_match = pred_counter == true_counter

    return tp, fp, fn, exact_match


def evaluate(model, tokenizer, test_data):
    results = []
    tp = fp = fn = 0
    exact_matches = 0
    valid_json = 0

    for i, example in enumerate(test_data):
        raw_output = generate_prediction(model, tokenizer, example)
        prediction = parse_prediction(raw_output)
        expected = example["label"]

        if prediction is not None:
            valid_json += 1

            sample_tp, sample_fp, sample_fn, exact = constraint_counts(
                prediction, expected
            )
        else:
            sample_tp = 0
            sample_fp = 0
            sample_fn = len(expected["constraints"])
            exact = False

        tp += sample_tp
        fp += sample_fp
        fn += sample_fn
        exact_matches += int(exact)

        results.append(
            {
                "text": example["text"],
                "reference_date": example["reference_date"],
                "expected": expected,
                "predicted": prediction,
                "raw_output": raw_output,
                "exact_match": exact,
            }
        )

        if (i + 1) % 25 == 0:
            print(f"Evaluated {i + 1}/{len(test_data)}")

    precision = tp / (tp + fp) if tp + fp else 0.0
    recall = tp / (tp + fn) if tp + fn else 0.0

    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    metrics = {
        "num_examples": len(test_data),
        "precision": precision,
        "recall": recall,
        "f1": f1,
        "exact_match_accuracy": exact_matches / len(test_data),
        "valid_json_rate": valid_json / len(test_data),
    }

    return metrics, results


def save_results(name, metrics, predictions):
    with open(OUTPUT_DIR / f"{name}_metrics.json", "w") as f:
        json.dump(metrics, f, indent=2)

    with open(OUTPUT_DIR / f"{name}_predictions.jsonl", "w") as f:
        for prediction in predictions:
            f.write(json.dumps(prediction) + "\n")


def main():
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    test_data = load_test_data(TEST_PATH)

    for name, fine_tuned in [
        ("base", False),
        ("finetuned", True),
    ]:
        print(f"\nEvaluating {name} model...")

        model = load_model(fine_tuned=fine_tuned)

        metrics, predictions = evaluate(model, tokenizer, test_data)

        save_results(f"{RUN_NAME}_{name}", metrics, predictions)

        print(f"\n{name.upper()} RESULTS")
        print(json.dumps(metrics, indent=2))

        # Release memory before loading the next model
        del model
        gc.collect()

        if DEVICE.type == "mps":
            torch.mps.empty_cache()


if __name__ == "__main__":
    main()
