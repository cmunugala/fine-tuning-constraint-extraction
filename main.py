import json
from datetime import datetime
from zoneinfo import ZoneInfo
from textwrap import dedent
from outlines import Generator, from_transformers


import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from models import *

model_name = "Qwen/Qwen2.5-0.5B-Instruct"
tokenizer = AutoTokenizer.from_pretrained(model_name)

model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
device = torch.device("mps" if torch.backends.mps.is_available() else "cpu")

model = model.to(device)
model.eval()

print(model.config.model_type)
print(f"Parameters: {model.num_parameters():,}")
print(f"Device: {next(model.parameters()).device}")

current_date = datetime.now(ZoneInfo("America/Los_Angeles")).date()

outlines_model = from_transformers(model, tokenizer)
generator = Generator(outlines_model, SchedulingOutput)

SYSTEM_PROMPT = f"""
You extract employee scheduling constraints.

Today's date is {current_date}.

Resolve relative dates into calendar dates.
Do not invent missing information.
Return scheduling constraints from the user's request.
""".strip()


def extract_constraints(user_message: str):
    current_date = datetime.now(ZoneInfo("America/Los_Angeles")).date()

    messages = [
        {"role": "system", "content": SYSTEM_PROMPT.format(current_date=current_date)},
        {"role": "user", "content": user_message},
    ]

    prompt = tokenizer.apply_chat_template(
        messages, tokenize=False, add_generation_prompt=True
    )

    response = generator(prompt, max_new_tokens=500)

    if isinstance(response, SchedulingOutput):
        return response

    return SchedulingOutput.model_validate_json(response)


result = extract_constraints("Sarah can't work Mondays after 4 pm.")

print(result.model_dump_json(indent=2))


schema = json.dumps(SchedulingOutput.model_json_schema(), indent=2)

messages = [
    {
        "role": "system",
        "content": (
            "Extract scheduling constraints. "
            "Return one constraint per rule mentioned. "
            "Never invent employees or rules. "
            "Return JSON."
        ),
    },
    {"role": "user", "content": "Sarah can't work Mondays after 4 pm."},
]

inputs = tokenizer.apply_chat_template(
    messages,
    add_generation_prompt=True,
    tokenize=True,
    return_dict=True,
    return_tensors="pt",
).to(device)

with torch.no_grad():
    outputs = model.generate(**inputs, max_new_tokens=150, do_sample=False)

response = tokenizer.decode(
    outputs[0][inputs["input_ids"].shape[1] :], skip_special_tokens=True
)

print(response)
