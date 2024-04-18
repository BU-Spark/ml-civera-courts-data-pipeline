import argparse
import os
from collections import defaultdict

import matplotlib.pyplot as plt
import networkx as nx
from tqdm import tqdm

from utils import load_json

parser = argparse.ArgumentParser()
parser.add_argument("--id", type=int, default=2, help="finetuned model id")
parser.add_argument("--limit", type=int, default=100, help="limit on data")
args = parser.parse_args()
model_id = args.id

base_path = "results/"
wrong_dir = os.path.join(base_path, f"wrong_Mistral_finetuned_model_id={model_id}_limit={args.limit}")

wrong_results = load_json(os.path.join(wrong_dir, "wrong_results.json"))

wrong_action_pair_count = defaultdict(int)
wrong_actor_pair_count = defaultdict(int)

for _id, res in (tq := tqdm(wrong_results.items())):
    gt = res["triplet"]["actor"]
    pre = res["answer"]["actor"]
    if gt != pre:
        wrong_actor_pair_count[(gt, pre)] += 1

    gt = res["triplet"]["action"]
    pre = res["answer"]["action"]
    if gt != pre:
        wrong_action_pair_count[(gt, pre)] += 1


def plot_bipartite_graph(pairs_dict, title):
    plt.clf()

    # 创建图形和轴对象
    fig, ax = plt.subplots(figsize=(10, 8))

    # 初始化网络图对象
    B = nx.Graph()

    # 添加节点与边
    left_nodes = set(pair[0] for pair in pairs_dict.keys())
    right_nodes = set(pair[1] for pair in pairs_dict.keys())

    print(len(left_nodes))
    print(len(right_nodes))

    B.add_nodes_from(left_nodes, bipartite=0)  # 添加左侧节点
    B.add_nodes_from(right_nodes, bipartite=1)  # 添加右侧节点

    counts = [count for (src, dst), count in pairs_dict.items()]
    max_count = max(counts)
    min_count = min(counts)

    # 添加带权重的边
    for (src, dst), count in pairs_dict.items():
        # B.add_edge(src, dst, weight=count)
        B.add_edge(src, dst, weight=(count - min_count) / max_count * 10 + 1)

    # 生成节点位置
    pos = dict()
    pos.update((node, (1, index)) for index, node in enumerate(left_nodes))
    pos.update((node, (2, index)) for index, node in enumerate(right_nodes))

    # 画节点
    nx.draw_networkx_nodes(B, pos, nodelist=left_nodes, node_color='r', node_size=100, ax=ax)
    nx.draw_networkx_nodes(B, pos, nodelist=right_nodes, node_color='b', node_size=100, ax=ax)

    # 画边
    for edge in B.edges(data=True):
        nx.draw_networkx_edges(B, pos, edgelist=[edge], width=edge[2]['weight'], ax=ax)

    # 标签
    labels = {}
    for node in B.nodes():
        labels[node] = node

    nx.draw_networkx_labels(B, pos, labels, font_size=10)

    # 设置图形标题和显示图形
    plt.title(title)
    plt.axis('off')  # 不显示坐标轴
    plt.tight_layout()
    # plt.show()

    plt.savefig(f"{wrong_dir}/{title}.png", dpi=300)


# 绘制图表
plot_bipartite_graph(wrong_actor_pair_count, "Actor Prediction Errors")
plot_bipartite_graph(wrong_action_pair_count, "Action Prediction Errors")
