import hashlib
import json

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


def execute_query(cursor, query):
    import time
    t = time.time()
    cursor.execute(query)
    print(f"{time.time() - t}s")


def get_hash_key(text):
    return hashlib.md5(text.encode('utf-8')).hexdigest()


def load_json(path):
    with open(path, "r", encoding="utf8") as f:
        return json.load(f)


def dump_json(obj, path, indent=None, **kwargs):
    with open(path, "w", encoding="utf8") as f:
        json.dump(obj, f, ensure_ascii=False, indent=indent, **kwargs)
