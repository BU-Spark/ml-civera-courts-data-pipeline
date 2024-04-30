"""
Author: Boyang Liu
Date: 2023-07-15

Description:
- Faiss Vector Database logic
- Store embeddings
- Text similarity search based on embeddings
"""


import os
from typing import Union

import numpy as np
from tqdm import tqdm

import faiss

from utils import dump_json, load_json
from embed import embed_desc


FAISS_MEASUREMENT_METRIC_DICT = {
    'cos': faiss.METRIC_INNER_PRODUCT,
    'l1': faiss.METRIC_L1,
    'l2': faiss.METRIC_L2,
    'l_inf': faiss.METRIC_Linf,
    'l_p': faiss.METRIC_Lp,
    'brayCurtis': faiss.METRIC_BrayCurtis,
    'canberra': faiss.METRIC_Canberra,
    'jensen_shannon': faiss.METRIC_JensenShannon
}


def get_measurement(measurement):
    return FAISS_MEASUREMENT_METRIC_DICT[measurement]


class DBFaiss:
    def __init__(self, limit, faiss_retrieval="HNSW64", faiss_measurement="cos"):
        """
        Initialize the FAISS database with specified limit, retrieval type, and measurement.

        Parameters:
        - limit (int): related to sampled data, influence the loaded and dumped filenames.
        - faiss_retrieval (str): The retrieval type to use with FAISS.
        - faiss_measurement (str): The distance measurement type.
        """

        self.dir_faiss_index = "embeddings/faiss_index"
        os.makedirs(self.dir_faiss_index, exist_ok=True)

        self.limit = limit
        index_name = f"limit={limit}"

        self.faiss_index_path = f"{self.dir_faiss_index}/{index_name}"
        self.faiss_keys_path = f"{self.dir_faiss_index}/{index_name}_keys.json"

        self.faiss_retrieval = faiss_retrieval
        self.faiss_measurement = faiss_measurement

    def save_embeddings(self, embeddings, keys):
        self.faiss_index = faiss.index_factory(
            len(embeddings[0]),
            self.faiss_retrieval,
            get_measurement(self.faiss_measurement)
        )
        assert self.faiss_index.ntotal == 0
        self.faiss_index.add(np.array(embeddings, dtype=np.float32))
        print(f"ITEM NUM: {self.faiss_index.ntotal}")
        self._save_faiss_index()

        self.faiss_idxes_to_keys = keys
        self.faiss_keys_to_idxes = {key: i for i, key in enumerate(keys)}
        self._save_faiss_keys()

    def load(self):
        self.load_faiss_index()
        self.load_faiss_keys()

    def _save_faiss_index(self):
        faiss.write_index(self.faiss_index, self.faiss_index_path)

    def _save_faiss_keys(self):
        dump_json(self.faiss_keys_to_idxes, self.faiss_keys_path)

    def clear_faiss(self):
        if os.path.isfile(self.faiss_index_path):
            os.remove(self.faiss_index_path)

    def load_faiss_index(self):
        self.faiss_index = faiss.read_index(self.faiss_index_path)
        assert self.faiss_index.is_trained, "Index dose not need to be trained"

    def load_faiss_keys(self):
        self.faiss_keys_to_idxes = load_json(self.faiss_keys_path)
        self.faiss_idxes_to_keys = list(self.faiss_keys_to_idxes.keys())

    def get_idx_of_key(self, key):
        return self.faiss_keys_to_idxes[key]

    def get_key_of_idx(self, idx):
        return self.faiss_idxes_to_keys[idx]

    def retrieve_vectors(self):
        embeddings, _ = embed_desc(self.limit)
        return np.array(embeddings)

    def get_idx_of_top_k_similar_results(self, query_embedding, top_k=5):
        query_embedding = np.array([query_embedding], dtype=np.float32)
        scores_or_distances, idxes = self.faiss_index.search(query_embedding, top_k)
        idxes = idxes[0]
        scores_or_distances = scores_or_distances[0]
        results_dict = [{
            'idx': idx,
            'score': float(score),
        } for idx, score in zip(idxes, scores_or_distances)]

        return results_dict

    def get_batch_embeddings(self, query_embeddings, top_k=5):
        """
        return: [top_k idxes, ... ], [top_k scores, ...]
        """

        query_embeddings = np.array(query_embeddings, dtype=np.float32)
        scores_or_distances_s, idxes_s = self.faiss_index.search(query_embeddings, top_k)

        return idxes_s, scores_or_distances_s
