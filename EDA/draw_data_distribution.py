import json
import matplotlib.pyplot as plt
import os


_dir = "distribution/"
os.makedirs(_dir, exist_ok=True)

TOPN = -1

# Load data from JSON file
with open('count/count.json', 'r') as file:
    data = json.load(file)

action_counts = data['action_counts']
actor_counts = data['actor_counts']

# Filter out empty action counts and corresponding names
action_items = [(name, int(value)) for name, value in action_counts.items() if name != ""]
action_items.sort(key=lambda x: x[1], reverse=True)
action_names, action_values = zip(*action_items)

# Filter out empty actor counts and corresponding names
actor_items = [(name, int(value)) for name, value in actor_counts.items() if name != ""]
actor_items.sort(key=lambda x: x[1], reverse=True)
actor_names, actor_values = zip(*actor_items)

non_empty_action_rows = sum(action_values)
action_rows = sum(data['action_counts'].values())
non_empty_actor_rows = sum(actor_values)
actor_rows = sum(data['actor_counts'].values())

print("non empty action rows:", non_empty_action_rows)
print("action rows:", action_rows)
print("non empty actor rows:", non_empty_actor_rows)
print("actor rows:", actor_rows)

assert action_rows == actor_rows

# Calculate non-empty ratio
non_empty_action_ratio = round(non_empty_action_rows / action_rows, 2) * 100
non_empty_actor_ratio = round(non_empty_actor_rows / actor_rows, 2) * 100

# Count number of action types
num_action_types = len(action_values)
# Count number of actor types
num_actor_types = len(actor_values)

# Plotting action counts
plt.figure(figsize=(10, 8))
if TOPN == -1:
    plt.barh(action_names, action_values)  # Adjust slice for visibility
else:
    plt.barh(action_names[:TOPN], action_values[:TOPN])  # Adjust slice for visibility
plt.xlabel('Counts')
plt.ylabel('Actions')
plt.title(f'Top 20 Legal Actions Distribution\nNumber of non-empty actions: {non_empty_action_rows}\nNon-empty ratio: {non_empty_action_ratio:.2f}%\nNumber of Action Types: {num_action_types}')
plt.tight_layout()

# Add count values next to each bar
for index, value in enumerate(action_values[:20]):
    plt.text(value, index, str(value), ha='left', va='center')

# plt.show()

if TOPN == -1:
    plt.savefig(_dir + 'legal_actions_distribution.png')
else:
    plt.savefig(_dir + f'top_{TOPN}_legal_actions_distribution.png')

# Plotting actor counts
plt.figure(figsize=(10, 8))
if TOPN == -1:
    plt.barh(actor_names, actor_values)  # Adjust slice for visibility
else:
    plt.barh(actor_names[:TOPN], actor_values[:TOPN])  # Adjust slice for visibility
plt.xlabel('Counts')
plt.ylabel('Actors')
plt.title(f'Top 20 Legal Actors Distribution\nNumber of non-empty actors: {non_empty_actor_rows}\nNon-empty ratio: {non_empty_actor_ratio:.2f}%\nNumber of Actor Types: {num_actor_types}')
plt.tight_layout()

# Add count values next to each bar
for index, value in enumerate(actor_values[:20]):
    plt.text(value, index, str(value), ha='left', va='center')

# plt.show()

if TOPN == -1:
    plt.savefig(_dir + 'legal_actors_distribution.png')
else:
    plt.savefig(_dir + f'top_{TOPN}_legal_actors_distribution.png')
