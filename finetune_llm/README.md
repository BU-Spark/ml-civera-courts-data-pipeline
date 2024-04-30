# Method: Finetune LLM

### Model
- Mistral-7B-Instruct-v0.2

### Prompts
```
@staticmethod
def prompt_triplet(triplet):
    prompt_template = (
        "[INST]"
        "Use precise words to summarize the 'actor' and 'action' of the description: {TRIPLET}\n"
        f"'actor' is an abstract entity in the legal case.\n"
        f"'action' is a summary of the behavior in legal events, which is the actor's action.\n"
        f"If 'actor' or 'action' is not empty, directly use it as the output."
        "Your output should be a json dict with two keys: 'actor' and 'action'."
        "[/INST]"
    )

    prompt = prompt_template.format(TRIPLET={"actor": "", "action": "", "description": triplet["description"]})
    answer = json.dumps({"actor": triplet["actor"], "action": triplet["action"]})

    return prompt, answer

@staticmethod
def formatting_func(prompt, answer):
    text = f"<s>{prompt}\n ### Answer: {answer}</s>"
    return text
```

## How to run
- Finetune
```
python finetune.py --limit 1000 --wandb
```
- Infer
```
python inference.py \
    --id {id of the finetuned model, in "models/{llm_name}/"} \
    --limit 1000
```
- Evaluate
```
python evaluate.py --id {model_id} --limit 1000
```
