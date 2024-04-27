"""
Author: Boyang Liu
Date: 2024-04-26

Description:
This script is used to:
- write to database
- use Predictor defined in predict.py
"""


import os

MODEL_CACHE_DIR = "/projectnb/ds549/students/liubyfly/models/"
os.environ["TRANSFORMERS_CACHE"] = MODEL_CACHE_DIR


import argparse
import time
import os

import pymysql
from tqdm import tqdm

from utils import *
from cfg.cfg import SRC_DB_NAME, DST_DB_NAME, TABLE_NAME, DIR_TEMP, total_records

from predict import Predictor


PATH_OFFSET = f"{DIR_TEMP}/offset.json"


def load_offset():
    with open(PATH_OFFSET, 'r') as file:
        return int(file.read().strip())


def dump_offset(offset: int):
    with open(PATH_OFFSET, 'w') as file:
        file.write(str(offset))


def add_columns_to_table():

    db_connection = get_connection(DST_DB_NAME)

    try:
        with db_connection.cursor() as cursor:
            # add new columns
            cursor.execute(f"SHOW COLUMNS FROM {TABLE_NAME} LIKE 'predicted_actor'")
            if not cursor.fetchone():  # If the column does not exist, then add it
                cursor.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN `predicted_actor` VARCHAR(255) DEFAULT NULL")

            cursor.execute(f"SHOW COLUMNS FROM {TABLE_NAME} LIKE 'predicted_action'")
            if not cursor.fetchone():  # If the column does not exist, then add it
                cursor.execute(f"ALTER TABLE {TABLE_NAME} ADD COLUMN `predicted_action` VARCHAR(255) DEFAULT NULL")

            db_connection.commit()
            print(f"Columns added to {TABLE_NAME} successfully.")
    except Exception as e:
        print(f"An error occurred while adding columns: {e}")
        db_connection.rollback()


def get_row_count(conn):
    """ Retrieve the number of rows in a specific table. """
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"SELECT COUNT(*) FROM {TABLE_NAME}")
            result = cursor.fetchone()
            count = result['COUNT(*)']
            print(count)
            return count  # Return the count of rows
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def get_columns(conn):
    """ Retrieve the column names of a specific table in MySQL. """
    try:
        with conn.cursor() as cursor:
            cursor.execute(f"SHOW COLUMNS FROM {TABLE_NAME}")
            columns = [column['Field'] for column in cursor.fetchall()]
            print(columns)
            return columns
    except Exception as e:
        print(f"An error occurred: {e}")
        return None


def write_predictions_to_db(cursor, predictions, if_update=True, if_tqdm=False):
    """ Write the prediction results into the output database. """
    if if_tqdm:
        predictions = tqdm(predictions)
    for i, pred in enumerate(predictions):
        if not if_tqdm:
            if i % 2000 == 0:
                print(f"{i}/{len(predictions)}")

        # Base query with placeholders for parameters
        sql = f"""INSERT INTO {TABLE_NAME} (case_action_id, description, actor, action)
                  VALUES (%s, %s, %s, %s)"""
        params = (pred['case_action_id'], pred['description'], pred['actor'], pred['action'])

        # Add the ON DUPLICATE KEY UPDATE clause if updates are allowed
        if if_update:
            sql += " ON DUPLICATE KEY UPDATE description = VALUES(description), actor = VALUES(actor), action = VALUES(action)"

        # print(sql)
        # Execute the query with parameters
        execute_query(cursor, sql, params=params)


def read_predict_write(offset, batch_size=10000, if_tqdm=False):
    input_conn = get_connection(SRC_DB_NAME)
    output_conn = get_connection(DST_DB_NAME)
    print("DB connected")

    predictor = Predictor()

    try:
        with input_conn.cursor() as input_cursor, output_conn.cursor() as output_cursor:

            # get_columns(output_conn)

            while True:

                start_t = time.time()

                print("\n", "=" * 10,
                      f"\033[91m{round(offset / total_records * 100, 3)}%\033[0m (offset={offset} / {total_records})",
                      "=" * 10, "\n")

                # Construct the query with dynamic batch size and offset
                query_sql = f"SELECT case_action_id, actor, action, description FROM {TABLE_NAME} LIMIT {batch_size} OFFSET {offset}"
                execute_query(input_cursor, query_sql, print_t=True)

                original_rows = input_cursor.fetchall()
                if not original_rows:  # Break the loop if no more data is fetched
                    break

                print("Predicting...")
                t = time.time()
                predicted_rows = predictor.predict(original_rows)  # Process data for prediction
                count_t(t, f"Predicted {batch_size} rows.")

                # print(len(original_rows))
                # print(len(predicted_rows))
                #
                # for predicted_row in predicted_rows:
                #     print(predicted_row)

                print("\nWriting...")
                t = time.time()
                # Write predicted results to the output database
                write_predictions_to_db(output_cursor, predicted_rows, if_tqdm=if_tqdm)
                output_conn.commit()  # Commit the transaction
                count_t(t, f"Wrote {batch_size} rows.")

                # print("count")
                # get_row_count(output_conn)

                offset += batch_size  # Update the offset for the next batch

                count_t(start_t, f"\nPredicted {offset} -> {offset + batch_size}")

                """ Dump offset """
                dump_offset(offset)

    except Exception as e:
        print(f"An error occurred: {e}")
        output_conn.rollback()  # Roll back in case of error

    finally:
        input_conn.close()
        output_conn.close()


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--add_columns", action="store_true", help="add new columns, direct change in the def")
    parser.add_argument("--tqdm", action="store_true")
    parser.add_argument("--first", action="store_true")
    parser.add_argument("--bz", type=int, default=10000)
    args = parser.parse_args()

    if args.add_columns:
        add_columns_to_table()
        return

    if args.first or not os.path.isfile(PATH_OFFSET):
        offset = 0
    else:
        offset = load_offset()

    read_predict_write(offset, if_tqdm=args.tqdm)


if __name__ == "__main__":
    main()
