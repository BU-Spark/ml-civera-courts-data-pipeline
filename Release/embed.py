"""
Author: Boyang Liu
Date: 2024-03-15

Description:
- Embedder
"""


import json
import time

import os
from typing import List, Dict

from tqdm import tqdm

from cfg.cfg import DIR_DATA, LIMIT
from utils import (
    get_hash_key,
    load_json,
    dump_json,
)

from sentence_transformers import SentenceTransformer


class Embedder:
    def __init__(self, load_model=True):
        self.dir_embedding = "embeddings/mlm_l6_v2"
        self.model_name = "all-MiniLM-L6-v2"

        for n, v in vars(self).items():
            if 'dir_' in n:
                os.makedirs(v, exist_ok=True)

        if load_model:
            t = time.time()
            self.embedding_model = SentenceTransformer(f'sentence-transformers/{self.model_name}')
            print(f"Loading model <{self.model_name}> takes {time.time() - t}s")

    def get_embedding(self, query: str, cache=False):
        if cache is False:
            return self.embedding_model.encode(query).tolist()
        # cache = true: store embedding for next usage
        # important for some apis
        # careful when there are lots of embeddings
        hk = get_hash_key(query)
        tmp_file = f"{self.dir_embedding}/{hk}.json"
        if os.path.isfile(tmp_file):
            try:
                return load_json(tmp_file)
            except:
                # Aborting the program suddenly can result in the file existing but being corrupted.
                pass
        embed = self.embedding_model.encode(query).tolist()
        dump_json(embed, tmp_file)
        return embed

    @staticmethod
    def batchify(iterable, n=1):
        le = len(iterable)
        for ndx in range(0, le, n):
            yield iterable[ndx: min(ndx + n, le)]

    # Function for batching and parallel processing the embeddings
    def embed(
            self,
            rows: Dict[str, Dict],
            tq_desc: str = "",
    ):
        """

        :param rows:
        :param tq_desc:
        :return: embeddings, keys
        """

        corpus = [row['description'] for row in rows.values()]
        # corpus = corpus[:100]

        embeddings = []
        for query in (tq := tqdm(corpus)):
            tq.set_description(tq_desc)
            embeddings.append(self.get_embedding(query))

        return embeddings, list(rows.keys())

    def embed_non_empty(
            self,
            rows: Dict[str, Dict],
            tq_desc: str = "",
            cache: bool = False,
    ):
        """

        :param rows:
        :param tq_desc:
        :param cache
        :return: embeddings, keys
        """

        non_empty_rows = {key: row for key, row in rows.items() if row["actor"] != "" and row["action"] != ""}

        corpus = [row['description'] for row in non_empty_rows.values()]
        # corpus = corpus[:100]

        embeddings = []
        for query in (tq := tqdm(corpus)):
            tq.set_description(tq_desc)
            embeddings.append(self.get_embedding(query, cache=cache))

        print(f"non_empty_rows={len(non_empty_rows)} / rows={len(rows)}.")

        return embeddings, list(non_empty_rows.keys())


def embed_desc(_limit=LIMIT):

    """
    sampled_data: {case_action_id: {actor, action, description}}

    :param _limit:
    :return:
    """

    with open(f"{DIR_DATA}/sampled_data_limit={_limit}.json", "r") as f:
        sampled_data = json.load(f)

    total_rows = len(sampled_data)
    print(total_rows)

    embedder = Embedder()
    # embeddings, keys = embedder.embed(sampled_data)
    # embeddings, keys = embedder.embed_non_empty(sampled_data)
    embeddings, keys = embedder.embed_non_empty(sampled_data, cache=False)

    return embeddings, keys


def main():
    from db_faiss import DBFaiss
    limit = LIMIT
    embeddings, keys = embed_desc(limit)

    db = DBFaiss(limit=limit)
    db.save_embeddings(embeddings, keys)


if __name__ == '__main__':
    main()
