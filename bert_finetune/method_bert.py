import argparse
import os
import random

from tqdm import tqdm
from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader, Subset
from transformers import BertTokenizer
import numpy as np

from torch.optim import AdamW

from utils import load_json


# cuda
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using {device} device")


def save_model_and_optimizer(model, optimizer, path="model_checkpoint.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, path)


def load_model_and_optimizer(model, optimizer, path="model_checkpoint.pth"):
    checkpoint = torch.load(path)
    model.load_state_dict(checkpoint['model_state_dict'])
    optimizer.load_state_dict(checkpoint['optimizer_state_dict'])


class BertForMultiTaskLearning(BertPreTrainedModel):
    def __init__(self, config, num_actors, num_actions):
        super().__init__(config)
        self.bert = BertModel(config)
        self.dropout = nn.Dropout(config.hidden_dropout_prob)

        # classifier for both tasks
        self.actor_classifier = nn.Linear(config.hidden_size, num_actors)
        self.action_classifier = nn.Linear(config.hidden_size, num_actions)

        print("init_weights")
        self.init_weights()

    def forward(self, input_ids, attention_mask=None):
        outputs = self.bert(input_ids, attention_mask=attention_mask)

        pooled_output = outputs[1]  # use cls output
        pooled_output = self.dropout(pooled_output)

        actor_logits = self.actor_classifier(pooled_output)
        action_logits = self.action_classifier(pooled_output)

        return actor_logits, action_logits


class CustomDataset(Dataset):
    def __init__(self, encodings, actor_labels, action_labels):
        self.encodings = encodings
        self.actor_labels = actor_labels
        self.action_labels = action_labels

    def __getitem__(self, idx):
        # item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        item['actor_labels'] = torch.tensor(self.actor_labels[idx], dtype=torch.long)
        item['action_labels'] = torch.tensor(self.action_labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.actor_labels)


def set_seed():
    seed = 42  # You can choose any number here
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

    # If you're using a multi-GPU setup, also set:
    torch.cuda.manual_seed_all(seed)


def get_actor_action_2_id():
    counts = load_json(f"../EDA/count/count.json")

    # actor
    actors = [k for k in counts["actor_counts"].keys() if k != ""]
    actors = sorted(actors)
    actor_2_id = {a: _id for _id, a in enumerate(actors) if a != ""}

    # action
    actions = [k for k in counts["action_counts"].keys() if k != ""]
    # ==================
    """
        # added new action
    """
    actions.append('No parties present')
    # ==================
    actions = sorted(actions)
    action_2_id = {a: _id for _id, a in enumerate(actions) if a != ""}

    # consider add 1 for new labels
    return actor_2_id, action_2_id


def prepare_data(_limit=1000):

    sample_data = load_json(f"../EDA/sample/sampled_data_limit={_limit}.json")

    empty_rows = {key: row for key, row in sample_data.items() if row["actor"] == "" or row["action"] == ""}
    non_empty_rows = {key: row for key, row in sample_data.items() if row["actor"] != "" and row["action"] != ""}

    # non_empty_rows = {k: v for i, (k, v) in enumerate(non_empty_rows.items()) if i < 3000}

    # key: action_id
    # value: row

    print(f"Empty: {len(empty_rows)}, Non-empty: {len(non_empty_rows)}")

    # non_empty_rows = {k: v for i, (k, v) in enumerate(non_empty_rows.items()) if i <= 10}

    # data
    descriptions = [row["description"] for row in non_empty_rows.values()]
    row_actors = [row["actor"] for row in non_empty_rows.values()]
    row_actions = [row["action"] for row in non_empty_rows.values()]

    # fixed map
    actor_2_id, action_2_id = get_actor_action_2_id()

    actors = [actor_2_id[actor] for actor in row_actors]
    actions = [action_2_id[action] for action in row_actions]

    tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
    encodings = tokenizer(descriptions, truncation=True, padding=True, return_tensors="pt")

    dataset = CustomDataset(encodings.data, actors, actions)

    # split dataset
    train_size = int(0.6 * len(dataset))
    val_size = int(0.1 * len(dataset))
    test_size = len(dataset) - train_size - val_size

    train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

    # # Create subsets for train, validation, and test
    # train_dataset = Subset(dataset, range(0, train_size))
    # val_dataset = Subset(dataset, range(train_size, train_size + val_size))
    # test_dataset = Subset(dataset, range(train_size + val_size, len(dataset)))

    # print(train_dataset[0])
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    """
    def check_labels(dataset):
        for data in dataset:
            actor_label = data['actor_labels']
            action_label = data['action_labels']
            assert 0 <= actor_label < len(actor_2_id), f"Invalid actor label {actor_label}"
            assert 0 <= action_label < len(action_2_id), f"Invalid action label {action_label}"

    check_labels(train_dataset)
    check_labels(val_dataset)
    check_labels(test_dataset)
    """

    return train_dataset, val_dataset, test_dataset, len(actor_2_id), len(action_2_id)


def train(model, train_dataset, val_dataset, epoch=5, save_dir="models/temp", if_tqdm=True):

    # init

    model.train()

    # param
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    optim = AdamW(model.parameters(), lr=5e-5)

    loss_fct = nn.CrossEntropyLoss()

    for epoch in range(epoch):
        if if_tqdm:
            train_loader = tqdm(train_loader)

        for step, batch in enumerate(train_loader):

            optim.zero_grad()
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            actor_labels = batch['actor_labels'].to(device)
            action_labels = batch['action_labels'].to(device)

            actor_logits, action_logits = model(input_ids=input_ids, attention_mask=attention_mask)

            # print(actor_logits)
            # print(action_labels)

            actor_loss = loss_fct(actor_logits, actor_labels)
            action_loss = loss_fct(action_logits, action_labels)
            total_loss = actor_loss + action_loss
            total_loss.backward()
            optim.step()

            if step % 100 == 0:
                print(f" Epoch={epoch}, Step={step}/{len(train_loader)} | Total Loss: {total_loss: .4f} - actor: {actor_loss:.4f}, action: {action_loss:.4f}")

        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        actor_accuracy, action_accuracy = evaluate_model(model, val_loader, if_tqdm=if_tqdm)
        print(f"Validation Accuracy - Actor: {actor_accuracy:.4f}, Action: {action_accuracy:.4f}")

        # save
        save_model_and_optimizer(model, optim, os.path.join(save_dir, f"checkpoint_{epoch}.pth"))

    return model


def evaluate_model(model, val_loader, if_tqdm=True):
    model.eval()
    correct_actor, correct_action = 0, 0
    total = 0

    with torch.no_grad():
        if if_tqdm:
            val_loader = tqdm(val_loader)

        for bi, batch in enumerate(val_loader):
            input_ids = batch['input_ids'].to(device)
            attention_mask = batch['attention_mask'].to(device)
            actor_labels = batch['actor_labels'].numpy()
            action_labels = batch['action_labels'].numpy()

            actor_logits, action_logits = model(input_ids=input_ids, attention_mask=attention_mask)
            actor_preds = torch.argmax(actor_logits, dim=1).cpu().numpy()
            action_preds = torch.argmax(action_logits, dim=1).cpu().numpy()

            correct_actor += (actor_preds == actor_labels).sum()
            correct_action += (action_preds == action_labels).sum()
            total += len(actor_labels)

            if bi % 100 == 0:
                print(f"{bi}/{len(val_loader)}")

    return correct_actor / total, correct_action / total


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)
    parser.add_argument("--epoch", type=int, default=1)
    parser.add_argument("--eval", action="store_true")
    parser.add_argument("--tqdm", action="store_true")

    args = parser.parse_args()

    save_dir = f"models/{args.limit}/"
    temp_dir = f"{save_dir}/temp"
    os.makedirs(temp_dir, exist_ok=True)
    final_model_path = f"{save_dir}/checkpoint_bert_{args.limit}"

    train_dataset, val_dataset, test_dataset, num_actors, num_actions = prepare_data(args.limit)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)

    if args.eval:

        model = BertForMultiTaskLearning.from_pretrained(
            final_model_path, num_actors=num_actors, num_actions=num_actions).to(device)

        actor_accuracy_test, action_accuracy_test = evaluate_model(model, test_loader, if_tqdm=args.tqdm)
        print(f"Test Accuracy - Actor: {actor_accuracy_test:.4f}, Action: {action_accuracy_test:.4f}")
        return

    model = BertForMultiTaskLearning.from_pretrained(
        'bert-base-uncased', num_actors=num_actors, num_actions=num_actions).to(device)

    train(model, train_dataset, val_dataset, args.epoch, save_dir=temp_dir, if_tqdm=args.tqdm)

    actor_accuracy_test, action_accuracy_test = evaluate_model(model, test_loader, if_tqdm=args.tqdm)
    print(f"Test Accuracy - Actor: {actor_accuracy_test:.4f}, Action: {action_accuracy_test:.4f}")

    model.save_pretrained(final_model_path)


if __name__ == '__main__':
    main()
