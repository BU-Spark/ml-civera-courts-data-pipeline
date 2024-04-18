import argparse
import os
from collections import defaultdict

from tqdm import tqdm

from utils import load_json, dump_json


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--id", type=int, default=2, help="finetuned model id")
    parser.add_argument("--limit", type=int, default=100, help="limit on data")
    args = parser.parse_args()
    model_id = args.id

    base_path = "results/"
    os.makedirs(base_path, exist_ok=True)
    wrong_dir = os.path.join(base_path, f"wrong_Mistral_finetuned_model_id={model_id}_limit={args.limit}")
    os.makedirs(wrong_dir, exist_ok=True)

    res_path = f"{base_path}Mistral_finetuned_model_id={model_id}_limit={args.limit}.json"
    results = load_json(res_path)

    count = actor_c = action_c = 0
    wrong_results = {}
    wrong_action_count = defaultdict(int)
    wrong_actor_count = defaultdict(int)

    for _id, res in (tq := tqdm(results.items())):
        count += 1
        correct = True

        # actor
        if res["triplet"]["actor"] == res["answer"]["actor"]:
            actor_c += 1
        else:
            correct = False
            wrong_actor_count[res["triplet"]["actor"]] += 1

        # action
        if res["triplet"]["action"] == res["answer"]["action"]:
            action_c += 1
        else:
            correct = False
            wrong_action_count[res["triplet"]["action"]] += 1

        tq.set_description(f"count={count} | actor_c={actor_c} | action_c={action_c}")

        if not correct:
            wrong_results[_id] = res

    # Save wrong actions and actors to separate JSON files
    dump_json(wrong_results, os.path.join(wrong_dir, "wrong_results.json"), indent=4)
    dump_json(dict(wrong_action_count), os.path.join(wrong_dir, "wrong_action_count.json"), indent=4)
    dump_json(dict(wrong_actor_count), os.path.join(wrong_dir, "wrong_actor_count.json"), indent=4)

    print(f"actor: {round(actor_c / count, 4)}")
    print(f"action: {round(action_c / count, 4)}")

    print(f"Wrong actor: {count - actor_c}")
    for k, v in wrong_actor_count.items():
        print(f"{k}: {v}")
    print()

    print(f"Wrong action: {count - action_c}")
    for k, v in wrong_action_count.items():
        print(f"{k}: {v}")


if __name__ == "__main__":
    main()
