"""
Author: Boyang Liu
Date: 2024-04-27

Description:
- Define bert predictor
- use model fine-tuned in finetune.py
"""


import torch.cuda
from torch.utils.data import DataLoader, Dataset
from tqdm import tqdm
from transformers import BertTokenizer

from finetune_bert import (
    BertForMultiTaskLearning,
    get_actor_action_2_id,
)
from cfg.cfg import ThresholdBert


class CiveraDatasetForInference(Dataset):
    def __init__(self, encodings):
        self.encodings = encodings
        self.length = self.encodings["input_ids"].shape[0]

    def __getitem__(self, idx):
        item = {key: val[idx].clone().detach() for key, val in self.encodings.items()}
        return item

    def __len__(self):
        return self.length


class BertPredictor:
    def __init__(self, batch_size=64):

        self.threshold = ThresholdBert

        self.batch_size = batch_size

        self.actor_2_id, self.action_2_id = get_actor_action_2_id()

        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        model_path = "./models/checkpoint_bert"
        self.bert_model = BertForMultiTaskLearning.from_pretrained(
            model_path, num_actors=len(self.actor_2_id), num_actions=len(self.action_2_id)
        ).to(self.device)
        self.bert_model.eval()

        self.tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')

    def predict(self, rows, if_tqdm=False):

        descriptions = [row['description'] for row in rows]

        encodings = self.tokenizer(descriptions, truncation=True, padding=True, return_tensors="pt")

        dataset = CiveraDatasetForInference(encodings.data)

        data_loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False)

        # print(len(dataset))
        # print(len(data_loader))

        # List to hold results that meet or exceed the threshold
        results_meet_threshold = []
        # List to hold results that do not meet the threshold
        results_below_threshold = []

        with torch.no_grad():
            if if_tqdm:
                data_loader = tqdm(data_loader)

            for bi, batch in enumerate(data_loader):
                if not if_tqdm:
                    if bi % 20 == 0:
                        print(f"{bi}/{len(data_loader)}")

                input_ids = batch['input_ids'].to(self.device)
                attention_mask = batch['attention_mask'].to(self.device)

                actor_logits, action_logits = self.bert_model(input_ids=input_ids, attention_mask=attention_mask)
                actor_preds = torch.argmax(actor_logits, dim=1).cpu().numpy()
                action_preds = torch.argmax(action_logits, dim=1).cpu().numpy()

                actor_max_probs = self._max_probability_from_logits(actor_logits)
                action_max_probs = self._max_probability_from_logits(action_logits)

                for i, row in enumerate(rows[bi * self.batch_size: (bi + 1) * self.batch_size]):
                    min_max_prob = min(actor_max_probs[i].item(), action_max_probs[i].item())
                    if min_max_prob >= self.threshold:
                        results_meet_threshold.append(
                            dict(
                                case_action_id=row["case_action_id"],
                                description=row["description"],
                                actor=actor_preds[i],
                                action=action_preds[i],
                                # method='bert',
                                # bert_actor_confidence=actor_max_probs[i],
                                # bert_action_confidence=action_max_probs[i],
                            )
                        )
                    else:
                        results_below_threshold.append(row)

        print(f"predicted={len(results_meet_threshold)}, rest={len(results_below_threshold)}")

        return results_meet_threshold, results_below_threshold

    @staticmethod
    def _max_probability_from_logits(logits):
        """
        Calculates the maximum softmax probability from logits for each prediction.

        Args:
        logits (torch.Tensor): The logits as output by a neural network model.

        Returns:
        torch.Tensor: The maximum probability for each prediction.
        """
        probabilities = torch.softmax(logits, dim=1)  # Convert logits to probabilities
        max_probabilities = torch.max(probabilities, dim=1)[0]  # Find the max probability for each prediction
        return max_probabilities

