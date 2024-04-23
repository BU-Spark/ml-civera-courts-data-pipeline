import argparse
import os

from tqdm import tqdm
from transformers import BertPreTrainedModel, BertModel
import torch
import torch.nn as nn

from torch.utils.data import Dataset, DataLoader
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
        item = {key: torch.tensor(val[idx]) for key, val in self.encodings.items()}
        item['actor_labels'] = torch.tensor(self.actor_labels[idx], dtype=torch.long)
        item['action_labels'] = torch.tensor(self.action_labels[idx], dtype=torch.long)
        return item

    def __len__(self):
        return len(self.actor_labels)


def prepare_data(_limit=100):

    sample_data = load_json(f"sample/sampled_data_limit={_limit}.json")

    empty_rows = {key: row for key, row in sample_data.items() if row["actor"] == "" or row["action"] == ""}
    non_empty_rows = {key: row for key, row in sample_data.items() if row["actor"] != "" and row["action"] != ""}

    # for i, row in enumerate(non_empty_rows.items()):
    #     if i % 100 != 0:
    #         continue
    #     print(row)
    # quit()

    # key: action_id
    # value: row

    print(f"Empty: {len(empty_rows)}, Non-empty: {len(non_empty_rows)}")

    # non_empty_rows = {k: v for i, (k, v) in enumerate(non_empty_rows.items()) if i <= 10}

    # data
    descriptions = [row["description"] for row in non_empty_rows.values()]
    row_actors = [row["actor"] for row in non_empty_rows.values()]
    row_actions = [row["action"] for row in non_empty_rows.values()]

    id_2_actor = list(set(row_actors))
    # consider add 1 for new labels
    actor_2_id = {actor: _id for _id, actor in enumerate(id_2_actor)}

    id_2_action = list(set(row_actions))
    # consider add 1 for new labels
    action_2_id = {action: _id for _id, action in enumerate(id_2_action)}

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

    print(type(train_dataset))
    # print(train_dataset[0])
    print(len(train_dataset), len(val_dataset), len(test_dataset))

    return train_dataset, val_dataset, test_dataset, len(id_2_actor), len(id_2_action)


def train(train_dataset, val_dataset, num_actors, num_actions, save_dir="models"):

    # init
    model = BertForMultiTaskLearning.from_pretrained(
        'bert-base-uncased', num_actors=num_actors, num_actions=num_actions).to(device)
    model.train()

    # param
    # train_loader = DataLoader(train_dataset, batch_size=2, shuffle=True)
    train_loader = DataLoader(train_dataset, batch_size=8, shuffle=True)
    optim = AdamW(model.parameters(), lr=5e-5)

    loss_fct = nn.CrossEntropyLoss()

    for epoch in range(10):
        for step, batch in enumerate((tq := tqdm(train_loader))):
            tq.set_description(f"Epoch: {epoch}")

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
                print(f" Step={step} | Total Loss: {total_loss: .4f} - actor: {actor_loss:.4f}, action: {action_loss:.4f}")

        val_loader = DataLoader(val_dataset, batch_size=16, shuffle=False)
        actor_accuracy, action_accuracy = evaluate_model(model, val_loader)
        print(f"Validation Accuracy - Actor: {actor_accuracy:.4f}, Action: {action_accuracy:.4f}")

        # save
        save_model_and_optimizer(model, optim, os.path.join(save_dir, f"checkpoint_{epoch}.pth"))

    return model


def evaluate_model(model, val_loader):
    model.eval()
    correct_actor, correct_action = 0, 0
    total = 0

    with torch.no_grad():
        for batch in val_loader:
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

    return correct_actor / total, correct_action / total


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--limit", type=int, default=100)

    args = parser.parse_args()

    train_dataset, val_dataset, test_dataset, num_actors, num_actions = prepare_data(args.limit)

    save_dir = f"models/{args.limit}/"
    os.makedirs(save_dir, exist_ok=True)

    model = train(train_dataset, val_dataset, num_actors, num_actions, save_dir=save_dir)

    test_loader = DataLoader(test_dataset, batch_size=16, shuffle=False)
    actor_accuracy_test, action_accuracy_test = evaluate_model(model, test_loader)
    print(f"Test Accuracy - Actor: {actor_accuracy_test:.4f}, Action: {action_accuracy_test:.4f}")


if __name__ == '__main__':
    main()
