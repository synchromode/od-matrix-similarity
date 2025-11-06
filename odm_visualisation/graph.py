import argparse

import matplotlib.pyplot as plt
import networkx as nx
import numpy as np

parser = argparse.ArgumentParser()
parser.add_argument("--file", required=True)
parser.add_argument("--hourly", action="store_true", default=False)
args = parser.parse_args()

od_matrix: np.ndarray = np.load(args.file)
od_matrix = np.sum(od_matrix, axis=0)

n_nodes = od_matrix.shape[0]

graph = nx.DiGraph()
for n in range(n_nodes):
    graph.add_node(n)
for o in range(n_nodes):
    for d in range(n_nodes):
        if o == d or od_matrix[o, d] == 0:
            continue
        graph.add_edge(o, d, weight=1/od_matrix[o, d])

pos = nx.spring_layout(graph, weight='weight')

nx.draw_networkx_nodes(graph, pos, node_size=10)

plt.show()