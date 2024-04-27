"""
Author: Boyang Liu
Date: 2024-04-26

Description:
- configs

- Limit is a key hyperparameter: defines the training data size
"""


import os

LIMIT = 1000

total_records = 73173625

# database
HOST = "masscourtsplus.org"
SRC_DB_NAME = "wp_courtdocs"
DST_DB_NAME = "wp_courtdocs_NORMALIZED"

TABLE_NAME = "cdocs_case_action_index"

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


# path
DIR_DATA = "data/"
DIR_SAMPLE = f"{DIR_DATA}/sample"
DIR_MODEL = "models/"
DIR_TEMP = "temp/"


# Automatically create directories for all variables starting with 'DIR_'
def create_directories():
    globals_dict = globals()  # Get the dictionary of global variables for the current module
    for var_name, path in globals_dict.items():
        if var_name.startswith("DIR_"):  # Check if the variable name starts with 'DIR_'
            os.makedirs(path, exist_ok=True)  # Create the directory, ignore if it already exists


create_directories()  # Call the function to execute directory creation


