"""
Author: Boyang Liu
Date: 2024-04-26

Description:
- Samples data from the database.

Noticing that action type distribution is average,
sample {LIMIT} data per action type.
"""


import argparse
import json
import os
import random
import time

from tqdm import tqdm

from utils import *
from cfg.cfg import LIMIT, SRC_DB_NAME, TABLE_NAME, DIR_DATA, DIR_SAMPLE


def get_sampled_data(_limit=LIMIT):
    """

    :param _limit:
    :return: dict: {action_type: {case_action_id, actor, action, description}}
    """

    connection = get_connection(SRC_DB_NAME)

    with open("cfg/labels.json", "r") as f:
        action_types = json.load(f)["action"]

    results = {}

    dir_case_action_ids = f"{DIR_SAMPLE}/case_action_ids/"
    os.makedirs(dir_case_action_ids, exist_ok=True)

    try:
        with connection.cursor() as cursor:
            """ 
            1. Extract primary key: case_action_id of each action type from database
            2. Sample {LIMIT} rows per action type
            """

            sample_case_action_ids_by_action_type = {}

            t = time.time()

            for _i, action_type in enumerate((tq := tqdm(action_types))):

                tq.set_description(f"Extracting Primary Keys... action type: <{action_type}>")

                # if action_type == "":
                #     continue

                # print("\n", "=" * 30)
                # print(f"action_type: <{action_type}>")

                try:
                    t = time.time()
                    result = load_json(f"{dir_case_action_ids}/{action_type}.json")
                    count_t(t, f"Load case_action_ids of <{action_type}>")

                # except FileNotFoundError:
                except:

                    query = f"""
                            SELECT case_action_id FROM {TABLE_NAME}
                            WHERE action = "{action_type}"
                            """

                    execute_query(cursor, query, print_t=False)

                    result = cursor.fetchall()
                    # ["case_action_id: ID},]
                    result = [ele["case_action_id"] for ele in result]
                    # print(len(result))
                    # print(result)

                    # t = time.time()
                    dump_json(result, f"{dir_case_action_ids}/{action_type}.json")
                    # count_t(t, f"Dump case_action_ids of <{action_type}>")

                # t = time.time()
                try:
                    sample_case_action_ids = random.sample(result, _limit)
                except ValueError:
                    print(len(result), _limit)
                    sample_case_action_ids = result
                # count_t(t, f"Sample {_limit} from {len(result)}")

                sample_case_action_ids_by_action_type[action_type] = sample_case_action_ids

            dump_json(sample_case_action_ids_by_action_type, f"{DIR_SAMPLE}/sampled_action2ids_limit={_limit}.json")

            count_t(t, f"fetch primary keys")

            results = {}

            t = time.time()
            for _i, (action_type, sample_case_action_ids) \
                    in enumerate(tq := tqdm(sample_case_action_ids_by_action_type.items())):

                tq.set_description(f"<{action_type}>")

                fetch_row_template = f"""
                    SELECT case_action_id, actor, action, description FROM {TABLE_NAME}
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

                execute_query(cursor, query, print_t=False)

                results[action_type] = cursor.fetchall()

            count_t(t, f"fetch sampled data limit = {_limit}")
            """"""
    except Exception as e:
        print(e)

    finally:
        connection.close()

    return results


def main():

    parser = argparse.ArgumentParser()
    parser.add_argument("--show", action="store_true")
    parser.add_argument("--limit", type=int, default=LIMIT)
    args = parser.parse_args()

    limit = args.limit

    if args.show:
        data = load_json(f"{DIR_SAMPLE}/sampled_data_based_on_action_type_limit={limit}.json")

        print(f"num of actions: {len(data)}")
        for action_type, action_data in data.items():
            print(f"{action_type}: {len(action_data)}")

        data_list = f"{DIR_DATA}/sampled_data_limit={limit}.json"
        print(f"Data Length: {len(data_list)}")

        print("Number of rows for action types")
        with open("cfg/labels.json", "r") as f:
            action_types = json.load(f)["action"]

        num_of_ids = {}
        for _i, action_type in enumerate((tq := tqdm(action_types))):
            dir_case_action_ids = f"{DIR_SAMPLE}/case_action_ids/"
            result = load_json(f"{dir_case_action_ids}/{action_type}.json")
            num_of_ids[action_type] = len(result)

        for action_type, n in num_of_ids.items():
            print(f"{action_type}: {n}")
        print("Total labeled action rows:", sum(num_of_ids.values()))
        return

    data = get_sampled_data(limit)

    with open(f"{DIR_SAMPLE}/sampled_data_based_on_action_type_limit={limit}.json", "w") as f:
        json.dump(data, f)

    print("Transforming... \n"
          "{action_type: [{case_action_id, actor, action, description}]} -> {case_action_id: {actor, action, description}}")
    # {action_type: {case_action_id, actor, action, description}} -> case_action_id: {actor, action, description}
    data_list = {}
    for action_type, rows in tqdm(data.items()):
        for row in rows:
            action_case_id = row.pop("case_action_id")  # remove 'action_case_id'
            data_list[action_case_id] = row

    dump_json(data_list, f"{DIR_DATA}/sampled_data_limit={limit}.json")


if __name__ == '__main__':
    main()
