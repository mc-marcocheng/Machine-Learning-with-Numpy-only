import matplotlib.pyplot as plt
import networkx as nx

G = nx.DiGraph(with_labels=False, rankdir="LR")
G.add_node(1, label="$x_1$", pos=(0, 2))
G.add_node(2, label="$x_2$", pos=(0, 1))
G.add_node(3, label="$x_3$", pos=(0, 0))
G.add_node(4, label="$x_1W_{11}+x_2W_{21}+x_3W_{31}+b_1$", pos=(1, 2))
G.add_node(5, label="$x_1W_{12}+x_2W_{22}+x_3W_{32}+b_2$", pos=(1, 1))
G.add_node(6, label="$x_1W_{13}+x_2W_{23}+x_3W_{33}+b_3$", pos=(1, 0))
G.add_node(7, label="softmax", pos=(2, 1))
G.add_node(8, label="$f_1$", pos=(3, 2))
G.add_node(9, label="$f_2$", pos=(3, 1))
G.add_node(10, label="$f_3$", pos=(3, 0))

for i in 1, 2, 3:
    for j in 4, 5, 6:
        G.add_edge(i, j, color="C1")
for i in 4, 5, 6:
    G.add_edge(i, 7, color="C2")
for i in 8, 9, 10:
    G.add_edge(7, i, color="C3")

nx.draw(
    G,
    pos={node: G.nodes[node]["pos"] for node in G.nodes()},
    with_labels=True,
    arrows=False,
    node_shape="s",
    node_color="none",
    edge_color=[G[u][v]["color"] for u, v in G.edges()],
    labels={node: G.nodes[node]["label"] for node in G.nodes()},
    bbox=dict(facecolor="C0", edgecolor="black", boxstyle="round,pad=0.2"),
)
plt.plot()
plt.savefig("assets/softmax_regression_networkx.png")
