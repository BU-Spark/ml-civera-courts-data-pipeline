import os
import random

from tqdm import tqdm

from utils import load_json, dump_json

import glm_inference

tokenizer, model = glm_inference.load()


_limit = 100

sample_data = load_json(f"sample/sampled_data_limit={_limit}.json")

empty_rows = {key: row for key, row in sample_data.items() if row["actor"] == "" or row["action"] == ""}
non_empty_rows = {key: row for key, row in sample_data.items() if row["actor"] != "" and row["action"] != ""}


def fill_row_template(row):
    row_template = "<description: {}, actor: {}, action: {}>\n"
    return row_template.format(row["description"], row["actor"], row["action"])


def parse_triplet(triplet_str):
    """
    Parses a string response containing actor and action information.

    :param triplet_str: A string formatted like "description: , actor: , action: "
    :return: A dictionary with the parsed 'actor' and 'action' values.
    """

    # Initialize result dictionary
    parsed_result = {"actor": None, "action": None}

    try:
        # Remove angle brackets and split the string
        cleaned_str = triplet_str.replace("<", "").replace(">", "")
        parts = cleaned_str.split(", ")

        # Iterate over parts to find and assign 'actor' and 'action'
        for part in parts:
            if part.startswith("actor:"):
                parsed_result["actor"] = part.replace("actor:", "").strip()
            elif part.startswith("action:"):
                parsed_result["action"] = part.replace("action:", "").strip()
    except Exception as e:
        print(f"Error parsing triplet: {e}")

    return parsed_result


prompt_template = (
    "Consider the examples provided below for guidance:\n"
    "{EXAMPLES}\n"
    "For the following description, accurately fill in any missing 'actor' or 'action' fields with a concise phrase, limited to 1-3 words. If 'actor' or 'action' is already provided, leave it as is without alteration:\n"
    "{TRIPLET}\n"
)

example_n = 5
prompt_example = ""
for i, row in enumerate(non_empty_rows.values()):
    if i == example_n:
        break
    # print(row)
    prompt_example += fill_row_template(row)
print(prompt_example)


test_n = 10
test_samples = random.sample(list(empty_rows.items()), test_n)


if_print = False

results = {}

for s in tqdm(test_samples):
    case_action_id, s = s

    # print(f"{case_action_id}: {s}")

    # actor = s["actor"]
    # action = s["action"]
    # description = s["description"]

    prompt = prompt_template.format(EXAMPLES=prompt_example, TRIPLET=fill_row_template(s))

    response = glm_inference.get_response(tokenizer, model, prompt)

    if if_print:
        print("============")
        print(s)
        print("Q:")
        print(prompt.removeprefix(prompt_example))
        print("A:")
        print(response)
        print()

    results[case_action_id] = {
        "ori": s,
        "response": parse_triplet(response),
    }


os.makedirs("llm", exist_ok=True)

dump_json(results, "llm/glm3_results.json", indent=4)
