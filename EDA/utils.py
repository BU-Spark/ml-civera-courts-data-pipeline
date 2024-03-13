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
