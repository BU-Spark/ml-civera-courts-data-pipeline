"""
Author: Boyang Liu
Date: 2024-04-26

Description:
- Define a predictor
- VDB -> Bert -> LLM
"""


import time

from utils import count_t
from vector_db_search import VectorDBSearchPredictor
from bert_inference import BertPredictor


class Predictor:
    def __init__(self):
        self.vdb_predictor = VectorDBSearchPredictor()
        self.bert_predictor = BertPredictor()

    def predict(self, rows):
        """ Apply predictive modeling to the fetched data from the database. """

        predicted_rows = []

        # 1. use vector db search method
        print("-" * 10, f"Predicting {len(rows)} rows by VDB...")
        t = time.time()
        temp_predicted_rows, rows = self.vdb_predictor.predict(rows)
        count_t(t, "VDB predictor")

        predicted_rows.extend(temp_predicted_rows)

        if len(rows) == 0:
            print("All solved by VDB")
            return predicted_rows

        # 2. use fine-tuned bert method
        print("-" * 10, f"Predicting {len(rows)} rows by Bert...")
        t = time.time()
        temp_predicted_rows, rows = self.bert_predictor.predict(rows)
        count_t(t, "Bert predictor")

        predicted_rows.extend(temp_predicted_rows)

        if len(rows) == 0:
            print("All solved by VDB + Bert")
            return predicted_rows

        # 3. use fine-tuned llm method
        print("-" * 10, f"Predicting {len(rows)} rows by LLM...")

        assert len(rows) == 0

        return predicted_rows


def predict_by_vector_db_search(rows):
    pass


def predict_by_bert(rows):
    pass


def predict_by_llm(rows):
    pass
