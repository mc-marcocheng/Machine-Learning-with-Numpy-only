import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph(with_labels=False, rankdir="LR")
G.add_node(1, pos=(0, 2))
G.add_node(2, pos=(0, 1))
G.add_node(3, pos=(0, 0))
G.add_node(4, pos=(1, 3))
G.add_node(5, pos=(1, 2))
G.add_node(6, pos=(1, 1))
G.add_node(7, pos=(1, 0))
G.add_node(8, pos=(2, 2))
G.add_node(9, pos=(2, 1))
G.add_node(10, pos=(2, 0))

for i in 1, 2, 3:
    for j in 4, 5, 6, 7:
        G.add_edge(i, j, color="C1")
for i in 4, 5, 6, 7:
    for j in 8, 9, 10:
        G.add_edge(i, 7, color="C2")

nx.draw(
    G,
    pos={node: G.nodes[node]["pos"] for node in G.nodes()},
    with_labels=True,
    arrows=False,
    node_shape="s",
    node_color="none",
    edge_color=[G[u][v]["color"] for u, v in G.edges()],
    bbox=dict(facecolor="C0", edgecolor="black", boxstyle="round,pad=0.2"),
)
plt.plot()
plt.savefig("assets/neural_network_networkx.png")
