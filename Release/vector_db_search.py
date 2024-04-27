"""
Author: Boyang Liu
Date: 2024-04-26

Description:
- Define vector database search predictor
- use embedder defined in embed.py
- use faiss db defined in db_faiss.py
"""


import numpy as np
from tqdm import tqdm

from embed import Embedder
from db_faiss import DBFaiss
import random

from cfg.cfg import LIMIT, TABLE_NAME, SRC_DB_NAME, DIR_DATA
from utils import (
    get_connection, execute_query,
    load_json, dump_json,
)


class VectorDBSearchPredictor:
    def __init__(self, _limit=LIMIT):
        self.embedder = Embedder(load_model=True)
        self.db = DBFaiss(limit=_limit)
        self.db.load()

        self.threshold = 0.8
        # self.threshold = 0.0
        self.top_k = 3

        # {case_action_id: {actor: , action:, desc: }
        self.vector_db_data = load_json(f"{DIR_DATA}/sampled_data_limit={_limit}.json")

    def get_embeddings(self, rows):
        # rows = tqdm(rows)
        return [self.embedder.get_embedding(row["description"]) for row in rows]

    def predict(self, rows):

        print("Embedding...")
        embeddings = self.get_embeddings(rows)

        print("Vector DB Searching...")
        idxes_s, scores_s = self.db.get_batch_embeddings(embeddings, top_k=self.top_k)

        """
        results: zip[(top_k idxes, top_k score), ...]
        """

        print("Postprocessing (filter by threshold)...")
        predicted_rows = []
        rest_rows = []

        for row, idxes, scores in zip(rows, idxes_s, scores_s):
            if np.mean(scores) >= self.threshold:
                reference_case_action_id = self.db.get_key_of_idx(idxes[0])
                reference = self.vector_db_data[reference_case_action_id]
                predicted_rows.append(
                    dict(
                        case_action_id=row["case_action_id"],
                        description=row["description"],
                        actor=reference["actor"],
                        action=reference["action"],
                        # method="vdb",
                        # vdb_confidence=np.mean(scores),
                    ))
            else:
                rest_rows.append(row)

        print(f"predicted={len(predicted_rows)}, rest={len(rest_rows)}")

        return predicted_rows, rest_rows


def main():
    _limit = LIMIT

    embedder = Embedder(load_model=True)
    db = DBFaiss(limit=_limit)
    db.load()

    sample_data = load_json(f"{DIR_DATA}/sampled_data_limit={_limit}.json")
    print(len(sample_data))

    empty_rows = {key: row for key, row in sample_data.items() if row["actor"] == "" or row["action"] == ""}
    non_empty_rows = {key: row for key, row in sample_data.items() if
                      row["actor"] != "" and row["action"] != ""}

    for desc, rows in {"empty": empty_rows, "non_empty": non_empty_rows}.items():

        print(desc)

        test_n = 10
        test_samples = random.sample(list(rows.items()), test_n)

        for s in test_samples:

            case_action_id, s = s

            print(f"case_action_id: {case_action_id} | test_sample: {s}")
            embedding = embedder.get_embedding(s["description"])
            topK_results = db.get_idx_of_top_k_similar_results(embedding, top_k=5)

            for idx_score in topK_results:
                case_action_id = db.get_key_of_idx(idx_score["idx"])
                res = non_empty_rows[case_action_id]
                print(res, idx_score["score"])

            print()


if __name__ == '__main__':
    main()
