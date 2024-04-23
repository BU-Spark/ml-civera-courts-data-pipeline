import os

MODEL_PATH = "/projectnb/ds549/students/liubyfly/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH

import json
from tqdm import tqdm
import re
import argparse
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel
from torch.utils.data import DataLoader, Dataset

from finetune import CiveraDataset
from utils import *
from load_data import load_data


def get_data(limit=100):
    id2triplet = load_data(_limit=limit)

    path = f"./data/id_limit={limit}.json"
    assert os.path.isfile(path)
    train_val_test = load_json(path)

    test_id2triplets = {_id: id2triplet[_id] for _id in train_val_test["test"]}

    return test_id2triplets


# Function to extract the numerical part of the checkpoint directory name
def extract_number(checkpoint_name):
    match = re.match(r'checkpoint-(\d+)', checkpoint_name)
    return int(match.group(1)) if match else -1


def load_model_tokenizer(model_id):

    lora_model_dir = f"/projectnb/ds549/students/liubyfly/models/mistral-civera-finetune/{model_id}/"

    # Get all the entries in the directory that match 'checkpoint-<number>'
    entries = os.listdir(lora_model_dir)
    checkpoint_dirs = [entry for entry in entries if re.match(r'checkpoint-\d+', entry)]

    # Find the checkpoint directory with the maximum number
    max_checkpoint_dir = max(checkpoint_dirs, key=extract_number, default=None)

    lora_model_path = os.path.join(lora_model_dir, max_checkpoint_dir)
    print("Lora model path:", lora_model_path)

    # quit()
    model_tokenizer_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_name_or_path, truncation=True)

    base_model = AutoModelForCausalLM.from_pretrained(model_tokenizer_name_or_path)
    ft_model = PeftModel.from_pretrained(base_model, lora_model_path)
    ft_model.eval()

    """
    The attention mask and the pad token id were not set. As a consequence, you may observe unexpected behavior. 
    Please pass your input's `attention_mask` to obtain reliable results.
    Setting `pad_token_id` to `eos_token_id`:2 for open-end generation.
    """
    # ft_model.config.pad_token_id = ft_model.config.eos_token_id
    # tokenizer.pad_token = tokenizer.eos_token

    return ft_model, tokenizer


def infer(ft_model, tokenizer, test_id2triplets, device, predictions, dump_path, if_print=False):

    print("all", len(test_id2triplets))
    test_id2triplets = {k: v for k, v in tqdm(test_id2triplets.items()) if k not in predictions}
    print("needed", len(test_id2triplets))

    with torch.no_grad():
        for i, (_id, triplet) in enumerate(test_id2triplets.items()):

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

            answer = decoded[0]

            # print(answer)

            answer = parse_answer(answer)

            predictions[_id] = {
                "triplet": triplet,
                "answer": answer,
            }

            if if_print:
                print()
                print(answer)
                print()
                print(triplet)
                print()

            print(f"{i}/{len(test_id2triplets)}")
            # if i >= 10:
            #     break

            if i % 100 == 0:
                dump_json(predictions, dump_path, indent=4)

    return predictions


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


def main():

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=3, help="finetuned model id")
    parser.add_argument("--limit", type=int, default=1000, help="finetuned model id")
    args = parser.parse_args()

    print(args)

    model_id = args.id
    limit = args.limit

    test_id2triplets = get_data(limit=limit)

    ft_model, tokenizer = load_model_tokenizer(model_id)
    ft_model.to(device)

    os.makedirs("results/", exist_ok=True)
    res_path = f"results/Mistral_finetuned_model_id={model_id}_limit={limit}.json"

    if_continue = True
    if if_continue and os.path.isfile(res_path):
        results = load_json(res_path)
    else:
        results = {}

    results = infer(ft_model, tokenizer, test_id2triplets, device, results, res_path, if_print=False)

    dump_json(results, res_path, indent=4)

    # results = load_json(f"results/Mistral_finetuned_model_id={model_id}_limit={limit}.json")
    # for result in results.values():
    #     parse_answer(result["answer"])


if __name__ == '__main__':
    main()
