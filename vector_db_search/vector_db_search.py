from embed import Embedder
from db_faiss import DBFaiss
import random

from utils import (
    get_connection, execute_query,
    load_json, dump_json,
)


def main():
    table_name = "cdocs_case_action_index"

    embedder = Embedder(load_model=True)
    db = DBFaiss(limit=100)
    db.load()

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            """"""

            query_sample_template = f"""
                      SELECT case_action_id, actor, action, description
                      FROM {table_name}
                      ?
                      """

            query_by_idx_template = f"""
                      SELECT case_action_id, actor, action, description
                      FROM {table_name}
                      WHERE case_action_id = ?
                      """

            #

            _limit = 100
            sample_data = load_json(f"sample/sampled_data_limit={_limit}.json")

            print(len(sample_data))
            # raise

            empty_rows = {key: row for key, row in sample_data.items() if row["actor"] == "" or row["action"] == ""}
            non_empty_rows = {key: row for key, row in sample_data.items() if
                              row["actor"] != "" and row["action"] != ""}

            for desc, rows in {"empty": empty_rows, "non_empty": non_empty_rows}.items():

                print(desc)

                test_n = 10
                test_samples = random.sample(list(rows.items()), test_n)

                for s in test_samples:

                    case_action_id, s = s

                    # query = query_sample_template.replace("?", f"")
                    # print(query)
                    # execute_query(cursor, query)
                    # s = cursor.fetchone()

                    print(f"case_action_id: {case_action_id} | test_sample: {s}")
                    embedding = embedder.get_embedding(s["description"])
                    topK_results = db.get_idx_of_top_k_similar_results(embedding, top_k=5)

                    for idx_score in topK_results:
                        case_action_id = db.get_key_of_idx(idx_score["idx"])
                        execute_query(cursor, query_by_idx_template.replace("?", str(case_action_id)))
                        res = cursor.fetchone()
                        print(res, idx_score["score"])

                    print()

    except Exception as e:
        print(e)

    finally:
        connection.close()


if __name__ == '__main__':
    main()
