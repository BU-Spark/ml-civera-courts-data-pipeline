import os
from typing import List

from tqdm import tqdm

from db_faiss import DBFaiss
from embed import Embedder
from utils import *


from glm_inference import get_response
# from gemini_inference import get_response


def load_data(limit=100):
    sample_data = load_json(f"sample/sampled_data_limit={limit}.json")

    empty_rows = {key: row for key, row in sample_data.items() if row["actor"] == "" or row["action"] == ""}
    non_empty_rows = {key: row for key, row in sample_data.items() if row["actor"] != "" and row["action"] != ""}

    return empty_rows, non_empty_rows


def get_embedding(embedder: Embedder, text: str):
    return embedder.get_embedding(query=text)


def get_references(connection: pymysql.connections.Connection, db: DBFaiss, embedding: List[float], top_k=3):

    table_name = "cdocs_case_action_index"
    query_by_idx_template = f"""
                          SELECT case_action_id, actor, action, description
                          FROM {table_name}
                          WHERE case_action_id = ?
                          """

    references = []

    with connection.cursor() as cursor:
        top_k_results = db.get_idx_of_top_k_similar_results(embedding, top_k)
        for idx_score in top_k_results:
            case_action_id = db.get_key_of_idx(idx_score["idx"])
            execute_query(cursor, query_by_idx_template.replace("?", str(case_action_id)))
            res = cursor.fetchone()
            res["score"] = round(idx_score["score"], 2)
            references.append(res)

    return references


def fill_row_template(row):
    row_template = "<actor: {}, action: {}, description: {}>"
    return row_template.format(row["actor"], row["action"], row["description"])


def fill_row_template_with_score(row):
    row_template = "<actor: {}, action: {}, description: {}, score: {}>"
    return row_template.format(row["actor"], row["action"], row["description"], row["score"])


def get_prompt(row, references):
    # Define a template for the prompt that will ask for predictions on missing actor and action.
    prompt_template = (
        "Your task is to use 1-3 words separately to predict the 'actor' and 'action' of the description: {DESCRIPTION}\n"
        "Actors are the roles or entities involved in legal actions, while actions are the specific legal steps or activities undertaken in legal proceedings.\n"
        "Here are some references for you:\n"
        "{REFERENCES}\n"
        "You need to pay attention to the score, which means the similarity between the task description and the reference description."
        "If the score is high (>0.8), you can directly use the the reference's actor and action to label the task description;"
        "If the score is low (<0.7), the references are almost useless, you need to generate the 'actor' and 'action' yourself based on the task description.\n"
        "Please output the prediction result as a json dict."
    )

    # references = [fill_row_template_with_score(ref) for ref in references]
    # qualified_references = [fill_row_template(ref) for ref in references if ref["score"] > 0.6]

    qualified_references = [fill_row_template_with_score(ref) for ref in references]

    # if len(qualified_references) == 0:
    #     qualified_references = [fill_row_template(references[0])]
    #     prompt_template.replace("Here are some references for you:", "Here is an example for you:")

    prompt = prompt_template.format(DESCRIPTION=row["description"], REFERENCES=qualified_references)
    return prompt


def predict_actor_action(prompt):
    response = get_response(prompt)

    # print(prompt)
    # print(response)

    try:
        return_dict = json.loads(response)
    except json.decoder.JSONDecodeError:
        try:
            response = response[response.index("{"): response.index("}") + 1]
            return_dict = json.loads(response)
        except json.decoder.JSONDecodeError:
            print(response)
            return_dict = None

    return return_dict


def main():

    exp_n = 100

    _limit = 100
    _top_k = 3

    embedder = Embedder()

    db = DBFaiss(limit=_limit)
    db.load()

    # tokenizer, model = glm_inference.load()

    connection = get_connection()
    empty_rows, non_empty_rows = load_data(limit=_limit)

    results = {}

    for i, (case_action_id, row) in enumerate(tqdm(empty_rows.items())):
    # for i, (case_action_id, row) in enumerate(empty_rows.items()):

        if exp_n != -1:
            if i >= exp_n:
                break

        embedding = get_embedding(embedder, row["description"])
        references = get_references(connection, db, embedding, top_k=_top_k)

        # for reference in references:
        #     print(reference)
        # continue

        prompt = get_prompt(row, references)

        actor_action_dict = predict_actor_action(prompt)

        if actor_action_dict is None:
            print(case_action_id)
            continue

        results[case_action_id] = {
            "row": row,
            "result": actor_action_dict,
            "references": references,
        }

    os.makedirs("results", exist_ok=True)
    dump_json(results, f"results/glm3_faiss_results_limit={_limit}.json", indent=4)


if __name__ == '__main__':
    main()
