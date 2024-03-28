import json
import os
import random
import time

from tqdm import tqdm

from utils import *

total_records = 73173625
table_name = "cdocs_case_action_index"


os.makedirs("sample", exist_ok=True)

def get_sampled_data(_limit=1000):
    """

    :param _limit:
    :return: dict: {action_type: {case_action_id, actor, action, description}}
    """

    with open("count/count.json", "r") as f:
        action_counts = json.load(f)["action_counts"]

    results = {}

    connection = get_connection()

    DIR_CASE_ACTION_IDS = "sample/case_action_ids/"
    os.makedirs(DIR_CASE_ACTION_IDS, exist_ok=True)

    try:
        with connection.cursor() as cursor:
            """"""

            sample_case_action_ids_by_action_type = {}

            for _i, action_type in enumerate(action_counts.keys()):

                # if action_type == "":
                #     continue

                print("\n==============")
                print(f"action_type: <{action_type}>")

                try:
                    t = time.time()
                    result = load_json(f"{DIR_CASE_ACTION_IDS}/{action_type}.json")
                    count_t(t, f"Load case_action_ids of <{action_type}>")

                # except FileNotFoundError:
                except:

                    query = f"""
                            SELECT case_action_id FROM {table_name}
                            WHERE action = "{action_type}"
                            """

                    execute_query(cursor, query, print_t=True)

                    result = cursor.fetchall()
                    # ["case_action_id: ID},]
                    result = [ele["case_action_id"] for ele in result]
                    print(len(result))

                    # print(result)

                    t = time.time()
                    dump_json(result, f"{DIR_CASE_ACTION_IDS}/{action_type}.json")
                    count_t(t, f"Dump case_action_ids of <{action_type}>")

                t = time.time()
                try:
                    sample_case_action_ids = random.sample(result, limit)
                except ValueError:
                    print(len(result), limit)
                    sample_case_action_ids = result
                count_t(t, f"Sample {limit} from {len(result)}")

                sample_case_action_ids_by_action_type[action_type] = sample_case_action_ids

            dump_json(sample_case_action_ids_by_action_type, f"sample/sampled_action2ids_limit={limit}.json")

            results = {}
            t = time.time()
            for _i, (action_type, sample_case_action_ids) \
                    in enumerate(tq := tqdm(sample_case_action_ids_by_action_type.items())):

                tq.set_description(f"<{action_type}>")

                fetch_row_template = f"""
                    SELECT case_action_id, actor, action, description FROM {table_name}
                    WHERE case_action_id = ?
                """

                # for case_action_id in sample_case_action_ids:
                #     # tq.set_description(str(case_action_id))
                #     execute_query(cursor, fetch_row_template.replace("?", str(case_action_id)))

                if len(sample_case_action_ids) == 1:
                    # in case of the situation like (100, ), will raise Error
                    query = fetch_row_template.replace("?", str(sample_case_action_ids[0]))
                else:
                    query = fetch_row_template.replace("= ?", f"in {tuple(sample_case_action_ids)}")
                # print(query)

                execute_query(cursor, query)

                results[action_type] = cursor.fetchall()

            count_t(t, f"fetch sampled data limit = {limit}")
            """"""
    except Exception as e:
        print(e)

    finally:
        connection.close()

    return results


limit = 100
data = get_sampled_data(limit)

with open(f"sample/sampled_data_based_on_action_type_limit={limit}.json", "w") as f:
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
