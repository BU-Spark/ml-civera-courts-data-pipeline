import json
import os

from tqdm import tqdm

from utils import get_connection, execute_query, dump_json

total_records = 73173625
table_name = "cdocs_case_action_index"


def get_sampled_data(_limit=1000):

    """

    :param _limit:
    :return: dict: {action_type: {case_action_id, actor, action, description}}
    """

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
                       SELECT case_action_id, actor, action, description
                       FROM {table_name}
                       WHERE action = "{action_type}"
                       LIMIT {_limit};
                       """
                # print(query)

                execute_query(cursor, query, False)

                results[action_type] = cursor.fetchall()

            """"""
    except Exception as e:
        print(e)

    finally:
        connection.close()

    return results


limit = 100
data = get_sampled_data(limit)


os.makedirs("sample", exist_ok=True)
with open(f"sample/sampled_data_based_on_actions_limit={limit}.json", "w") as f:
    json.dump(data, f)


print("Transforming... \n"
      "{action_type: [{case_action_id, actor, action, description}]} -> {case_action_id: {actor, action, description}}")
# {action_type: {case_action_id, actor, action, description}} -> case_action_id: {actor, action, description}
data_list = {}
for action_type, rows in tqdm(data.items()):
    for row in rows:
        action_case_id = row.pop("case_action_id")  # remove 'action_case_id'
        data_list[action_case_id] = row

dump_json(data_list, f"sample/sampled_data_limit={limit}.json")
