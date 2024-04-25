import argparse
import os
from collections import defaultdict
from tqdm import tqdm

from utils import load_json, dump_json


def main():
    # Set up argument parser
    parser = argparse.ArgumentParser(
        description="Analyze the performance of a finetuned model based on triplet prediction results.")
    parser.add_argument("--id", type=int, default=3, help="Finetuned model ID.")
    parser.add_argument("--limit", type=int, default=1000, help="Maximum number of data points to process.")
    args = parser.parse_args()

    # Create directories for output
    base_path = "results/"
    os.makedirs(base_path, exist_ok=True)
    wrong_dir = os.path.join(base_path, f"wrong_Mistral_finetuned_model_id={args.id}_limit={args.limit}")
    os.makedirs(wrong_dir, exist_ok=True)

    # Load results from a specific model and limit
    res_path = f"{base_path}Mistral_finetuned_model_id={args.id}_limit={args.limit}.json"
    results = load_json(res_path)

    # Initialize counters
    count = actor_c = action_c = 0
    gen_actor_c = gen_action_c = 0
    wrong_results = {}
    wrong_action_count = defaultdict(int)
    wrong_actor_count = defaultdict(int)
    generated_answers = {'actor': {}, 'action': {}}

    # Load domain data
    action_actor_count = load_json("../EDA/count/count.json")
    actor_domain = set(action_actor_count["actor_counts"].keys())
    action_domain = set(action_actor_count["action_counts"].keys())
    print(f"Actor Domain Size: {len(actor_domain)}, Action Domain Size: {len(action_domain)}")

    # Process each result
    for _id, res in tqdm(results.items(), desc="Processing results"):
        try:
            count += 1
            correct = True

            # Check and process actor results
            actor = res["triplet"]["actor"]
            predicted_actor = res["answer"]["actor"]
            if predicted_actor == actor:
                actor_c += 1
            else:
                correct = False
                wrong_actor_count[actor] += 1
                if predicted_actor not in actor_domain:
                    generated_answers['actor'][_id] = res
                    gen_actor_c += 1

            # Check and process action results
            action = res["triplet"]["action"]
            predicted_action = res["answer"]["action"]
            if predicted_action == action:
                action_c += 1
            else:
                correct = False
                wrong_action_count[action] += 1
                if predicted_action not in action_domain:
                    generated_answers['action'][_id] = res
                    gen_action_c += 1

            if not correct:
                wrong_results[_id] = res
        except Exception as e:
            print(f"Error processing result {_id}: {e}")

    # Save incorrect results
    dump_json(wrong_results, os.path.join(wrong_dir, "wrong_results.json"), indent=4)
    dump_json(dict(wrong_actor_count), os.path.join(wrong_dir, "wrong_actor_count.json"), indent=4)
    dump_json(dict(wrong_action_count), os.path.join(wrong_dir, "wrong_action_count.json"), indent=4)
    dump_json(dict(generated_answers), os.path.join(wrong_dir, "generated_answers.json"), indent=4)

    # Output statistics
    print(f"Actor accuracy: {actor_c / count:.4f}")
    print(f"Action accuracy: {action_c / count:.4f}")
    print(f"Generated Actor: {gen_actor_c / count:.4f}")
    print(f"Generated Action: {gen_action_c / count:.4f}")

    # print(f"Wrong actor: {count - actor_c}")
    # for k, v in wrong_actor_count.items():
    #     print(f"{k}: {v}")
    #
    # print(f"Wrong action: {count - action_c}")
    # for k, v in wrong_action_count.items():
    #     print(f"{k}: {v}")


if __name__ == "__main__":
    main()
