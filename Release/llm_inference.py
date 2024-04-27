"""
Author: Boyang Liu
Date: 2024-04-27

Description:
- Infer by llm: mistral
- Use lora fine-tuned in funetune_llm.py
"""


import os

MODEL_CACHE_DIR = "/projectnb/ds549/students/liubyfly/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR

import json
from tqdm import tqdm
import re
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset, default_collate

from utils import *
from cfg.cfg import LIMIT, DIR_MODEL


class InferenceDataset(Dataset):
    def __init__(self, tokenizer, id2rows):
        self.tokenizer = tokenizer

        self.id2rows = id2rows
        self.ids = list(id2rows.keys())

        self.message_content_template = (
            "Use precise words to summarize the 'actor' and 'action' of the description: "
            f"{{'actor': '', 'action': '', 'description': _DESCRIPTION_}}\n"
            f"'actor' is an abstract entity in the legal case.\n"
            f"'action' is a summary of the behavior in legal events, which is the actor's action.\n"
            f"If 'actor' or 'action' is not empty, directly use it as the output."
            "Your output should be a json dict with two keys: 'actor' and 'action'."
        )

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        case_action_id = self.ids[idx]
        message = [{
            "role": "user",
            "content": self.message_content_template.replace(
                "_DESCRIPTION_", self.id2rows[case_action_id]["description"])
        }]
        # encodes = self.tokenizer.apply_chat_template(message, return_tensors="pt", padding=True, max_length=512)

        prompt = self.tokenizer.apply_chat_template(message, tokenize=False)

        return dict(
            case_action_id=case_action_id,
            prompt=prompt,
        )

    @staticmethod
    def collate_fn(batch):
        batch_ids = [item['case_action_id'] for item in batch]
        batch_prompts = [item['prompt'] for item in batch]

        return {
            'case_action_ids': batch_ids,
            'prompts': batch_prompts,
        }


class LLMPredictor:
    def __init__(self, batch_size=16):

        self.batch_size = batch_size
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        self.base_model_tokenizer_name = "mistralai/Mistral-7B-Instruct-v0.2"
        self.lora_model_path = f"{DIR_MODEL}/lora_mistral/checkpoint-2501"

        self.tokenizer = AutoTokenizer.from_pretrained(self.base_model_tokenizer_name)
        self.tokenizer.pad_token = self.tokenizer.eos_token

        base_model = AutoModelForCausalLM.from_pretrained(self.base_model_tokenizer_name)
        self.ft_model = PeftModel.from_pretrained(base_model, self.lora_model_path).to(self.device)
        self.ft_model.eval()

    def predict(self, rows, if_tqdm=False):

        id2rows = {row["case_action_id"]: row for row in rows}

        dataset = InferenceDataset(self.tokenizer, id2rows)

        data_loader = DataLoader(dataset, batch_size=self.batch_size, num_workers=16, shuffle=False,
                                 collate_fn=InferenceDataset.collate_fn)

        predictions = []
        rest_rows = []

        with torch.no_grad():
            if if_tqdm:
                data_loader = tqdm(data_loader)

            for i, batch in enumerate(data_loader):

                ids = batch["case_action_ids"]
                prompts = batch["prompts"]

                inputs = self.tokenizer(prompts, padding="max_length", max_length=512, return_tensors="pt")
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                generated_ids = self.ft_model.generate(**inputs, max_new_tokens=256, do_sample=True,
                                                       pad_token_id=self.tokenizer.eos_token_id)

                decoded = self.tokenizer.batch_decode(generated_ids)

                for case_action_id, answer in zip(ids, decoded):

                    answer = self.parse_answer(answer)

                    predictions.append(dict(
                        case_action_id=case_action_id,
                        description=id2rows[case_action_id]["description"],
                        actor=answer["actor"],
                        action=answer["action"],
                    ))

                if not if_tqdm:
                    # if i % 100 == 0:
                    print(f"{i + 1} / {len(data_loader)}")

        return predictions, rest_rows

    @staticmethod
    def parse_answer(answer):
        # Regular expression to find JSON object after "### Answer:"
        pattern = r'### Answer:\s*(\{.*?\})'

        # Using re.findall to extract all matches
        matches = re.findall(pattern, answer, re.DOTALL)

        # for match in matches:
        #     print(match)

        for match in matches:
            res_str = match
            break
        else:
            # no match
            return answer

        try:
            res = json.loads(res_str)
            return res
        except Exception as e:
            print(e)
            # cannot be parsed into json dict
            return answer
