import json
import time
import os

import pymysql
from tqdm import tqdm

# Load database connection information from a file
with open('cfg/account.yml', 'r') as file:
    lines = file.readlines()
    user = lines[0].strip()
    password = lines[1].strip()

# Database connection information
host = "masscourtsplus.org"  # Replace with your actual host address
dbname = "wp_courtdocs"  # The database you are connecting to

# List of table names for reference
table_names = [
    "cdocs_case_action_index",
    "cdocs_case_meta_index",
    "cdocs_party_assignment_index",
    "cdocs_party_index"
]

# Select the table to work with
table_name = "cdocs_case_action_index"

# Connect to the database
connection = pymysql.connect(host=host,
                             user=user,
                             password=password,
                             database=dbname,
                             charset='utf8mb4',
                             cursorclass=pymysql.cursors.DictCursor)

action_counts = {}
actor_counts = {}

try:
    with connection.cursor() as cursor:
        """ count the number of different actions and actors """
        # Count the number of different actions
        t = time.time()
        query_action = "SELECT action, COUNT(*) AS count FROM cdocs_case_action_index GROUP BY action"
        cursor.execute(query_action)
        print(f"{time.time() - t}s")
        for row in tqdm(cursor.fetchall()):
            action_counts[row['action']] = row['count']

        # Count the number of different actors
        t = time.time()
        query_actor = "SELECT actor, COUNT(*) AS count FROM cdocs_case_action_index GROUP BY actor"
        cursor.execute(query_actor)
        print(f"{time.time() - t}s")
        for row in tqdm(cursor.fetchall()):
            actor_counts[row['actor']] = row['count']

finally:
    # Ensure the database connection is closed
    connection.close()

print("Action counts:", action_counts)
print("Actor counts:", actor_counts)

# Combine the counts into one dictionary for easier storage
statistics = {
    "action_counts": action_counts,
    "actor_counts": actor_counts
}


os.makedirs("count", exist_ok=True)

# Save the statistics to 'count.json'
with open('count/count.json', 'w') as file:
    json.dump(statistics, file, ensure_ascii=False, indent=4)
