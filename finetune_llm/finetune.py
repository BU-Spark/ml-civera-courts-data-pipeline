import argparse
import os
MODEL_PATH = "/projectnb/ds549/students/liubyfly/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_PATH

from datetime import datetime

import time
import json
import torch

import transformers
from sklearn.model_selection import train_test_split

from torch.utils.data import DataLoader, Dataset
from transformers import AutoModelForCausalLM, AutoTokenizer
from peft import get_peft_model, LoraConfig, TaskType, prepare_model_for_kbit_training

from load_data import load_data
from utils import *


# Example dataset
class CiveraDataset(Dataset):
    def __init__(self, tokenizer, triplets_list, max_length=1024):
        self.tokenizer = tokenizer
        self.triplets = triplets_list
        self.max_length = max_length

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

    def __len__(self):
        return len(self.triplets)

    def __getitem__(self, idx):
        prompt, answer = self.prompt_triplet(self.triplets[idx])
        # return self.tokenizer(self.formatting_func(prompt, answer))
        return self.tokenizer(self.formatting_func(prompt, answer))


def prepare_model(model):
    model.gradient_checkpointing_enable()
    model = prepare_model_for_kbit_training(model)

    config = LoraConfig(
        r=32,
        lora_alpha=64,
        target_modules=[
            "q_proj",
            "k_proj",
            "v_proj",
            "o_proj",
            "gate_proj",
            "up_proj",
            "down_proj",
            "lm_head",
        ],
        bias="none",
        lora_dropout=0.05,  # Conventional
        task_type="CAUSAL_LM",
    )

    model = get_peft_model(model, config)
    print_trainable_parameters(model)
    return model


def accelerate_model(model):
    from accelerate import FullyShardedDataParallelPlugin, Accelerator
    from torch.distributed.fsdp.fully_sharded_data_parallel import FullOptimStateDictConfig, FullStateDictConfig

    fsdp_plugin = FullyShardedDataParallelPlugin(
        state_dict_config=FullStateDictConfig(offload_to_cpu=True, rank0_only=False),
        optim_state_dict_config=FullOptimStateDictConfig(offload_to_cpu=True, rank0_only=False),
    )

    accelerator = Accelerator(fsdp_plugin=fsdp_plugin)

    if torch.cuda.device_count() > 1:  # If more than 1 GPU
        model.is_parallelizable = True
        model.model_parallel = True

    model = accelerator.prepare_model(model)

    return model


def train(model, tokenizer, tokenized_train_dataset, tokenized_val_dataset, report_to=None):

    project = "civera-finetune"
    base_model_name = "mistral"
    run_name = base_model_name + "-" + project
    output_dir = f"{MODEL_PATH}/" + run_name

    _i = 0
    while True:
        if os.path.isdir(output_dir + f"/{_i}"):
            _i += 1
            continue
        output_dir = output_dir + f"/{_i}"
        break
    print(output_dir)

    if report_to == "wandb":
        print("Report To Wandb")
        import wandb
        wandb.login()

        wandb_project = f"civera-finetune-model_id={_i}"
        if len(wandb_project) > 0:
            os.environ["WANDB_PROJECT"] = wandb_project

    trainer = transformers.Trainer(
        model=model,
        train_dataset=tokenized_train_dataset,
        eval_dataset=tokenized_val_dataset,
        args=transformers.TrainingArguments(
            output_dir=output_dir,
            warmup_steps=1,  # Number of steps to perform learning rate warmup
            per_device_train_batch_size=2,  # Batch size per device during training
            gradient_accumulation_steps=10,  # Number of steps to accumulate gradients before updating weights
            gradient_checkpointing=True,  # Enable gradient checkpointing to save memory
            num_train_epochs=5,  # Set the number of training epochs
            learning_rate=2.5e-5,  # Set a small learning rate for fine-tuning
            bf16=True,  # Enable bfloat16 mixed precision training
            optim="paged_adamw_8bit",  # Use the 8-bit version of AdamW optimizer for efficient memory usage
            # logging_strategy="epoch",  # Log metrics at the end of each epoch
            logging_strategy="steps",
            logging_steps=25,
            logging_dir=f"{output_dir}/logs",  # Directory where logs will be stored
            save_strategy="epoch",  # Save model checkpoints at the end of each epoch
            evaluation_strategy="epoch",  # Evaluate the model at the end of each epoch
            eval_steps=None,  # No need to set eval_steps since the strategy is based on epochs
            do_eval=True,  # Perform evaluation at the end of each epoch
            report_to=report_to,  # "wandb"  # Uncomment to use Weights & Biases for monitoring, if desired
            run_name=f"{run_name}-{datetime.now().strftime('%Y-%m-%d-%H-%M')}",  # Name the training run
            disable_tqdm=True,
        ),
        data_collator=transformers.DataCollatorForLanguageModeling(tokenizer, mlm=False),
    )

    model.config.use_cache = False  # silence the warnings. Please re-enable for inference!
    trainer.train()


def print_max_length(tokenizer, triplets_list):
    # Tokenize each text and count the number of tokens
    token_lengths = [len(tokenizer.encode(CiveraDataset.prompt_triplet(triplet), add_special_tokens=True))
                     for triplet in triplets_list]

    # Calculate maximum length
    max_length = max(token_lengths)

    # Calculate percentiles
    import numpy as np
    percentile_90 = int(np.percentile(token_lengths, 90))
    percentile_80 = int(np.percentile(token_lengths, 80))
    percentile_50 = int(np.percentile(token_lengths, 50))

    # Print out the lengths
    print(f"Maximum length: {max_length}")
    print(f"90th percentile length: {percentile_90}")
    print(f"80th percentile length: {percentile_80}")
    print(f"50th percentile length: {percentile_50}")


def print_trainable_parameters(model):
    """
    Prints the number of trainable parameters in the model.
    """
    trainable_params = 0
    all_param = 0
    for _, param in model.named_parameters():
        all_param += param.numel()
        if param.requires_grad:
            trainable_params += param.numel()
    print(
        f"trainable params: {trainable_params} || all params: {all_param} || trainable%: {100 * trainable_params / all_param}"
    )


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", default=100, type=int)
    parser.add_argument("--wandb", action="store_true")
    args = parser.parse_args()
    print(args)
    print("=" * 30)
    print()

    model_tokenizer_name_or_path = "mistralai/Mistral-7B-Instruct-v0.2"

    tokenizer = AutoTokenizer.from_pretrained(model_tokenizer_name_or_path)
    tokenizer.pad_token = tokenizer.eos_token

    # dataset
    limit = args.limit
    id2triplet = load_data(_limit=limit)

    os.makedirs("./data", exist_ok=True)
    path = f"./data/id_limit={limit}.json"
    if os.path.isfile(path):
        print("Loading ids")
        train_val_test = load_json(path)

    else:
        # Split data into train, validation, and test
        # 6:2:2
        _train_val_ids, test_ids = train_test_split(list(id2triplet.keys()), test_size=1 / 5, random_state=42)
        train_ids, val_ids = train_test_split(_train_val_ids, test_size=1 / 4, random_state=42)

        train_val_test = {
            "train": train_ids,
            "val": val_ids,
            "test": test_ids,
        }

        print("Dumping ids")
        dump_json(train_val_test, path)

    train_triplets = [id2triplet[_id] for _id in train_val_test["train"]]
    val_triplets = [id2triplet[_id] for _id in train_val_test["val"]]
    test_triplets = [id2triplet[_id] for _id in train_val_test["test"]]

    train_dataset = CiveraDataset(tokenizer, train_triplets)
    val_dataset = CiveraDataset(tokenizer, val_triplets)
    test_dataset = CiveraDataset(tokenizer, test_triplets)

    print(len(train_dataset), len(val_dataset))
    # quit()

    # model
    model = AutoModelForCausalLM.from_pretrained(model_tokenizer_name_or_path)
    model = prepare_model(model)

    # accelerate
    # model = accelerate_model(model)

    start_t = time.time()

    report_to = None
    if args.wandb:
        report_to = "wandb"

    # train
    train(model, tokenizer, train_dataset, val_dataset, report_to=report_to)

    count_t(start_t, "Train")


if __name__ == '__main__':
    main()
