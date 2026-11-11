import json

with open("../experiments/baseline_base_predictions.jsonl") as f:
    predictions = [json.loads(line) for line in f]

for example in predictions[:5]:
    print("INPUT:", example["text"])
    print("EXPECTED:", example["expected"])
    print("PREDICTED:", example["predicted"])
    print()


with open("../experiments/baseline_finetuned_predictions.jsonl") as f:
    predictions = [json.loads(line) for line in f]

for example in predictions[:5]:
    print("INPUT:", example["text"])
    print("EXPECTED:", example["expected"])
    print("PREDICTED:", example["predicted"])
    print()
