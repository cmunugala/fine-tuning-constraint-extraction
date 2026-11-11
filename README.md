# Fine Tuning 

Fine-tuned a small language model to extract structured employee scheduling constraints from natural-language requests.Started with **Qwen2.5-0.5B-Instruct** as the base model and then fine-tuned using **LoRA**. 

Project Rationale: Building an agent to perform scheduling for a highly-regulated company (e.g medical offices) may require an LLM hosted locally. For that reason, I am exploring the ability of SLM's to extract structured scheduling constraints from natural language text. 

## Results

Fine-tuning substantially improved structured constraint extraction performance compared with the prompted base model. Note: Results are currently on validation set. Held-out test set is being reserved for the time being in case additional experiments are performed. 

| Metric | Base Model | Fine-Tuned Model |
|---|---:|---:|
| Strict constraint-level F1 | 0.0% | **92.5%** |
| Request-level exact-match accuracy | 0.0% | **89.1%** |
| Parseable JSON rate | 94.9% | **100%** |

Evaluated on 175 held-out synthetic validation examples using the same system prompt and generation settings.

**Evaluation methodology:**

- **Strict constraint-level F1:** A predicted constraint is considered correct only when all its fields match a ground-truth constraint. True positives, false positives, and false negatives are aggregated across the validation set to calculate micro-averaged F1.
- **Request-level exact-match accuracy:** A request is correct only when every ground-truth constraint is extracted accurately, with no missing or additional constraints.
- **Parseable JSON rate:** Percentage of outputs that can be parsed into the expected top-level JSON structure after removing optional Markdown code fences.


## Example: Base Model vs. Fine-Tuned Model

**Input:**

> FYI: Camille: max 7 days straight.

### Expected Output

```json
{
  "constraints": [
    {
      "employee": "Camille",
      "type": "max_consecutive_days",
      "priority": "hard",
      "value": 7.0
    }
  ]
}
```

### Base Model Output

```json
{
  "constraints": [
    {
      "employee": "Camille",
      "type": "max_weekly_hours",
      "min_weekly_hours": 7,
      "priority": "hard",
      "value": 7.0,
      "time_scope": {
        "days_of_week": [
          "Monday",
          "Tuesday",
          "Wednesday",
          "Thursday",
          "Friday"
        ],
        "start_time": "09:00:00",
        "end_time": "17:00:00",
        "start_date": "2028-06-05",
        "end_date": "2028-06-07"
      }
    }
  ]
}
```

The base model incorrectly classified the constraint as `max_weekly_hours`, introduced an unsupported field, and hallucinated working hours, weekdays, and dates that were never mentioned.

The fine-tuned model matched the expected output exactly.