"""
Author: Boyang Liu
Date: 2024-04-26

Description:
- Store actor types and action types.
- Count actor types and action types.
"""


import time
import os

import pymysql
from tqdm import tqdm

from utils import *
from cfg.cfg import TABLE_NAME, SRC_DB_NAME


def count_actor_action():
    connection = get_connection(SRC_DB_NAME)

    action_counts = {}
    actor_counts = {}

    try:
        with connection.cursor() as cursor:
            """ count the number of different actions and actors """

            # Count the number of different actors
            query_actor = f"SELECT actor, COUNT(*) AS count FROM {TABLE_NAME} GROUP BY actor"

            execute_query(cursor, query_actor, print_t=True)
            for row in tqdm(cursor.fetchall()):
                actor_counts[row['actor']] = row['count']

            # Count the number of different actions
            query_action = f"SELECT action, COUNT(*) AS count FROM {TABLE_NAME} GROUP BY action"
            execute_query(cursor, query_action, print_t=True)

            for row in tqdm(cursor.fetchall()):
                action_counts[row['action']] = row['count']

    finally:
        # Ensure the database connection is closed
        connection.close()

    return actor_counts, action_counts


def store_actor_action_counts(actor_counts, action_counts):

    # labels
    actors = sorted([k for k in actor_counts.keys() if k != ""])
    actions = sorted([k for k in action_counts.keys() if k != ""])
    labels = {
        "actor": actors,
        "action": actions,
    }
    print(len(actors), len(actions))

    dump_json(labels, "cfg/labels.json", indent=4)


    # counts
    # print("Action counts:", action_counts)
    # print("Actor counts:", actor_counts)

    # Combine the counts into one dictionary for easier storage
    statistics = {
        "action_counts": action_counts,
        "actor_counts": actor_counts
    }
    os.makedirs("data", exist_ok=True)
    dump_json(statistics, "data/count.json", indent=4)


def main():
    actor_counts, action_counts = count_actor_action()
    store_actor_action_counts(actor_counts, action_counts)


if __name__ == '__main__':
    main()
