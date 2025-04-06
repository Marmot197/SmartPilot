import numpy as np
import pandas as pd
import networkx as nx
from pyvis.network import Network
from causallearn.search.FCMBased import lingam
import os

# ✅ Node Descriptions and Colors (same as in modified_diffan.py)
NODE_DESCRIPTIONS = {
    
}

NODE_COLORS = {
    "Sensor": "#1f77b4",   # Blue
    "Actuator": "#ff7f0e", # Orange
    "Unknown": "#d62728"   # Red
}


def plot_lingam_causal_graph(adj_matrix, node_labels, filename="lingam_causal_graph.html"):
    """Creates an interactive causal graph with Pyvis and embedded node legends."""
    G = nx.DiGraph()

    # Add nodes and edges
    for i in range(len(node_labels)):
        G.add_node(node_labels[i])

    for i in range(len(adj_matrix)):
        for j in range(len(adj_matrix[i])):
            if adj_matrix[i, j] != 0:
                G.add_edge(node_labels[i], node_labels[j])

    # Create Pyvis Network
    net = Network(height="800px", width="80%", directed=True, notebook=True)
    net.toggle_physics(False)

    # Add nodes with color and description
    for node in G.nodes:
        node_type, desc = NODE_DESCRIPTIONS.get(node, ("Unknown", "No description available."))
        node_color = NODE_COLORS.get(node_type, NODE_COLORS["Unknown"])
        net.add_node(node, label=node, title=desc, color=node_color, size=20, physics=False)

    # Add edges
    for edge in G.edges():
        src, dst = edge
        net.add_edge(src, dst, title=f"{src} → {dst}", color="gray")

    # Save basic graph first
    net.save_graph(filename)

    # Inject custom HTML for legends
    with open(filename, "r", encoding="utf-8") as f:
        html_content = f.read()

    # Legend for descriptions (Right)
    legend_html = """
    <div style="position: fixed; top: 50px; right: 20px; width: 300px; background-color: white;
                padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px gray;
                font-family: Arial, sans-serif; overflow-y: auto; max-height: 80vh;">
        <h4 style="margin: 0; padding-bottom: 10px;">Node Descriptions</h4>
        <ul style="list-style: none; padding: 0; margin: 0;">
    """
    for node, (_, desc) in NODE_DESCRIPTIONS.items():
        legend_html += f"<li><strong>{node}</strong>: {desc}</li>"

    legend_html += """
        </ul>
    </div>
    """

    # Legend for types (Left)
    node_type_legend = """
    <div style="position: fixed; top: 50px; left: 20px; width: 200px; background-color: white;
                padding: 15px; border-radius: 10px; box-shadow: 2px 2px 10px gray;
                font-family: Arial, sans-serif;">
        <h4 style="margin: 0; padding-bottom: 10px;">Node Types</h4>
        <ul style="list-style: none; padding: 0; margin: 0;">
    """
    for typ, color in NODE_COLORS.items():
        node_type_legend += f"""<li style="color: {color}; font-weight: bold;">● {typ}</li>"""

    node_type_legend += """
        </ul>
    </div>
    """

    # Append legends before </body>
    html_content = html_content.replace("</body>", legend_html + node_type_legend + "</body>")

    with open(filename, "w", encoding="utf-8") as f:
        f.write(html_content)

    print(f"Graph saved as {filename}. Open in a browser to view.")


# ✅ Load dataset dynamically
csv_file = "uploaded_dataset.csv"
df = pd.read_csv(csv_file, usecols=["I_R01_Gripper_Load", "I_R01_Gripper_Pot", "I_R02_Gripper_Load", "I_R02_Gripper_Pot", "I_R03_Gripper_Load", "I_R03_Gripper_Pot", "I_R04_Gripper_Load", "I_R04_Gripper_Pot"])  # dynamically modified to use specific columns
data = df.head(1000).to_numpy()
node_labels = df.columns.tolist()

# ✅ Run LiNGAM
print("\n### Running LiNGAM for Causal Discovery ###")
model = lingam.ICALiNGAM()
model.fit(data)

# ✅ Extract adjacency matrix
adj_matrix = model.adjacency_matrix_

# ✅ Plot interactive causal graph
plot_lingam_causal_graph(adj_matrix, node_labels, filename="lingam_causal_graph.html")

import networkx as nx
import pickle
G = nx.DiGraph()


# After model.fit(data) and graph is created
G = nx.DiGraph()
for i in range(len(adj_matrix)):
    for j in range(len(adj_matrix[i])):
        if adj_matrix[i, j] != 0:
            G.add_edge(node_labels[i], node_labels[j])

# Save edges
with open("lingam_graph_edges.pkl", "wb") as f:
    pickle.dump(list(G.edges()), f)

# ✅ Save adjacency matrix and column order for causal reasoning queries
with open("lingam_adjacency_matrix.pkl", "wb") as f:
    pickle.dump((adj_matrix, node_labels), f)