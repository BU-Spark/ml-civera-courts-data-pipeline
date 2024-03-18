import json
import os

from tqdm import tqdm

from utils import get_connection, execute_query

total_records = 73173625
table_name = "cdocs_case_action_index"


def get_sampled_data(_limit=10000):

    with open("count/count.json", "r") as f:
        action_counts = json.load(f)["action_counts"]

    results = {}

    connection = get_connection()

    try:
        with connection.cursor() as cursor:
            """"""

            # query = f"SELECT actor, action, description, case_id FROM {table_name} WHERE action = 'Case disposed'"
            # query = f"SELECT actor, action, description, case_id FROM {table_name} WHERE action = ''"
            # query = f"SELECT actor, action, description, case_id FROM {table_name} WHERE action = 'Affidavit filed'"

            for _i, action_type in enumerate(tq := tqdm(action_counts.keys())):
                tq.set_description(f"{action_type} | {(_i+1/len(action_counts))}")

                query = f"""
                       SELECT case_id, actor, action, description
                       FROM {table_name}
                       WHERE action = "{action_type}"
                       LIMIT {_limit};
                       """
                # print(query)

                execute_query(cursor, query)

                results[action_type] = cursor.fetchall()

            """"""
    except Exception as e:
        print(e)

    finally:
        connection.close()

    return results


limit = 10000
data = get_sampled_data(limit)


os.makedirs("sample", exist_ok=True)
with open(f"sample/sampled_data_based_on_actions_limit={limit}.json", "w") as f:
    json.dump(data, f)

