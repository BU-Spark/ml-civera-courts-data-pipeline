import argparse
import os

from tqdm import tqdm

from utils import load_json, dump_json

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=2, help="finetuned model id")
parser.add_argument("--limit", type=int, default=100, help="finetuned model id")
args = parser.parse_args()
model_id = args.id

os.makedirs("results/", exist_ok=True)
res_path = f"results/Mistral_finetuned_model_id={model_id}_limit={args.limit}.json"

results = load_json(res_path)


count = actor_c = action_c = 0

wrong = {}

for _id, res in (tq := tqdm(results.items())):

    count += 1

    correct = True
    # actor
    if res["triplet"]["actor"] == res["answer"]["actor"]:
        actor_c += 1
    else:
        correct = False

    if res["triplet"]["action"] == res["answer"]["action"]:
        action_c += 1
    else:
        correct = False

    tq.set_description(f"count={count} | actor_c={actor_c} | action_c={action_c}")

    if not correct:
        wrong[_id] = res
        print(res)


print(f"actor: {round(actor_c / count, 4)}")
print(f"action: {round(action_c / count, 4)}")


dump_json(wrong, res_path.replace("results/", "results/wrong_"), indent=4)