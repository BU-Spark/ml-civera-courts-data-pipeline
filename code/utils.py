import hashlib
import json
import time

import pymysql


def get_connection():
    # Load database connection information from a file
    with open('cfg/account.yml', 'r') as file:
        lines = file.readlines()
        user = lines[0].strip()
        password = lines[1].strip()

    # Database connection information
    host = "masscourtsplus.org"  # Replace with your actual host address
    dbname = "wp_courtdocs"  # The database you are connecting to

    """
    # List of table names for reference
    table_names = [
        "cdocs_case_action_index",
        "cdocs_case_meta_index",
        "cdocs_party_assignment_index",
        "cdocs_party_index"
    ]

    # Select the table to work with
    table_name = "cdocs_case_action_index"
    """

    # Connect to the database
    connection = pymysql.connect(host=host,
                                 user=user,
                                 password=password,
                                 database=dbname,
                                 charset='utf8mb4',
                                 cursorclass=pymysql.cursors.DictCursor)

    return connection


def execute_query(cursor, query, print_t=False):
    import time
    t = time.time()
    cursor.execute(query)
    if print_t:
        count_t(t, f"Executing query:\n{query}\n")


def get_hash_key(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def dump_json(obj, path, indent=None, **kwargs):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, **kwargs)


def convert_seconds_to_hms(seconds):

    if seconds < 1:
        return f"{round(seconds, 4)}s"

    seconds = int(seconds)
    hours = seconds // 3600
    minutes = (seconds % 3600) // 60
    seconds = seconds % 60
    return f"{hours:02d}:{minutes:02d}:{seconds:02d}"


def count_t(start_t, desc="..."):
    dt = time.time() - start_t
    print_t = convert_seconds_to_hms(dt)
    print(f"{desc} takes {print_t}.")


def run_and_count_t(f, *args, desc="..."):
    print(f"{desc} starts")
    t = time.time()
    f(*args)
    dt = time.time() - t
    print_t = convert_seconds_to_hms(dt)
    print(f"{desc} takes {print_t}.")
