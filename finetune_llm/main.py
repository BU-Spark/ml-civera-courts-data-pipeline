import json
import os
import random

from tqdm import tqdm
from utils import load_json, dump_json
from load_data import load_data

from inference import infer


limit = 100

data = load_data(_limit=limit)

aa = load_json("../EDA/count/count.json")

actors = list(aa["actor_counts"].keys())
actions = list(aa["action_counts"].keys())


exp_n = 5
samples = random.sample(list(data.items()), exp_n)
examples = [triplet for s, triplet in samples]
output_examples = [f'example{i + 1}: {{"actor": {triplet["actor"]}, "action": {triplet["action"]}}}'
                   for i, (_, triplet) in enumerate(samples)]

prompt_template = (
    "Your task is use fewer than 3 words to summarize the 'actor' and 'action' of the description: _DESCRIPTION_\n"
    f"Actor is an abstract entity in the legal case, "
    f"you must choose actor from the actor domain: {actors}.\n"
    f"Action is a summary of the behavior in legal events, action should be related to actor, "
    f"you must choose action from the action domain: {actions}.\n"
    "Here are some examples for you:\n"
    f"{examples}\n"
    "Your output should be a json dict, and the keys are 'actor' and 'action'.\n"
    f"Here are some output examples for you: " + '\n'.join(output_examples)
)

print(prompt_template)


results = {}
for i, (s, triplet) in enumerate(tqdm(data.items())):

    if i % 1000 != 0:
        continue

    # print(s, triplet)
    prompt = prompt_template.replace("_DESCRIPTION_", triplet["description"])
    response = infer(prompt)

    res_str = response.split("[/INST]")[-1]
    res_str = res_str.strip().strip("</s>")

    try:
        res = json.loads(res_str)
    except:
        print("Error:", s, res_str)
        continue

    print()
    print("=" * 30)
    print(triplet)
    print(res)
    print("=" * 30)
    print()
    continue

    results[s] = {
        "case": triplet,
        "res": res,
    }

    # print(response)
    # print()

    # if i >= 10:
    #     break

quit()

os.makedirs("results/", exist_ok=True)

dump_json(results, f"results/Mistral_direct_limit={limit}.json", indent=4)
