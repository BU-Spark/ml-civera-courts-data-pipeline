import os

import torch

MODEL_PATH = "/projectnb/ds549/projects/civera/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH

from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader

from finetune import CiveraDataset
from utils import *
from load_data import load_data

limit = 100

id2triplet = load_data(_limit=limit)

path = f"./data/id_limit={limit}.json"
assert os.path.isfile(path)
train_val_test = load_json(path)

test_triplets = [id2triplet[_id] for _id in train_val_test["test"]]

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

lora_model_path = "/projectnb/ds549/students/liubyfly/models/mistral-civera-finetune/0/checkpoint-285/"
model_tokenizer_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"

tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_name_or_path)

base_model = AutoModelForCausalLM.from_pretrained(model_tokenizer_name_or_path)
ft_model = PeftModel.from_pretrained(base_model, lora_model_path)
ft_model.eval()
ft_model.to(device)

predictions = []
with torch.no_grad():
    for triplet in test_triplets:

        messages = [
            {"role": "user",
             "content":
                "Use precise words to summarize the 'actor' and 'action' of the description: "
                f"{{'actor': '', 'action': '', 'description':{triplet['description']}}}\n"
                f"'actor' is an abstract entity in the legal case.\n"
                f"'action' is a summary of the behavior in legal events, which is the actor's action.\n"
                f"If 'actor' or 'action' is not empty, directly use it as the output."
                "Your output should be a json dict with two keys: 'actor' and 'action'."},
        ]
        encodeds = tokenizer.apply_chat_template(messages, return_tensors="pt")

        model_inputs = encodeds.to(device)

        generated_ids = ft_model.generate(model_inputs, max_new_tokens=256, do_sample=True)
        decoded = tokenizer.batch_decode(generated_ids)
        print()
        print(decoded[0])
        print(triplet)
        print()

