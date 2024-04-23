import random

from utils import load_json


def load_data(_limit=100):
    sample_data = load_json(f"../EDA/sample/sampled_data_limit={_limit}.json")

    empty_rows = {key: row for key, row in sample_data.items() if row["actor"] == "" or row["action"] == ""}
    non_empty_rows = {key: row for key, row in sample_data.items() if row["actor"] != "" and row["action"] != ""}

    return non_empty_rows

    for desc, rows in {"empty": empty_rows, "non_empty": non_empty_rows}.items():
        print(desc)

        test_n = 10
        test_samples = random.sample(list(rows.items()), test_n)

        for s in test_samples:
            case_action_id, s = s

            print(f"{case_action_id}: {s}")

            actor = s["actor"]
            action = s["action"]
            description = s["description"]

            """
            Your method here
            """