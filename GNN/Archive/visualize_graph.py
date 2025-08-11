import torch
import matplotlib.pyplot as plt
import networkx as nx
from torch_geometric.utils import to_networkx
import numpy as np

# --- Step 1: Load the graph ---
graph_path = "data/graphs/horizontal_fibers_graph.pt"
try:
    # Load the graph data. weights_only=False is required for loading graph objects.
    data = torch.load(graph_path, weights_only=False)
except FileNotFoundError:
    print(f"Error: Graph file not found at {graph_path}")
    print("Please ensure you have run the updated 'mesh_to_graph.py' script first.")
    exit()
except Exception as e:
    print(f"An error occurred while loading the graph: {e}")
    exit()

# --- Step 2: Print debug info to verify the graph ---
print(f"Graph loaded successfully from: {graph_path}")
print("-" * 30)
print(f"Number of nodes: {data.num_nodes}")
print(f"Number of edges: {data.num_edges}")
print(f"Node feature shape: {data.x.shape}") # Should be [num_nodes, 5]
print(f"Edge feature shape: {data.edge_attr.shape if data.edge_attr is not None else 'N/A'}")
print("-" * 30)

# --- Step 3: Set up graph properties for visualization ---
# Get node positions for the layout from the 'pos' attribute
pos = {i: data.pos[i].tolist() for i in range(data.num_nodes)}

# Convert the PyG Data object to a NetworkX graph for drawing
G = to_networkx(data, to_undirected=True)

# --- Plot 1: Visualization by Boundary Condition ---
# Assign colors based on Boundary Condition ID (now the 5th column, index 4)
if data.x.shape[1] == 5:
    bc_ids = data.x[:, 4].numpy()
    bc_color_map = {0: '#3498db', 1: '#e67e22', 2: '#2ecc71'} # Blue, Orange, Green
    bc_node_colors = [bc_color_map.get(bc_id, '#bdc3c7') for bc_id in bc_ids]
    
    # Create legend for boundary conditions
    bc_legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label='Interior (BC=0)', markersize=10, markerfacecolor=bc_color_map.get(0)),
        plt.Line2D([0], [0], marker='o', color='w', label='Fixed (BC=1)', markersize=10, markerfacecolor=bc_color_map.get(1)),
        plt.Line2D([0], [0], marker='o', color='w', label='Strained (BC=2)', markersize=10, markerfacecolor=bc_color_map.get(2))
    ]
else:
    print("Warning: Node features do not have the expected shape of 5. Cannot plot boundary conditions.")
    bc_node_colors = '#bdc3c7'
    bc_legend_handles = []

# Draw the boundary condition graph
plt.figure(figsize=(10, 10))
plt.title("Graph Visualization by Boundary Condition", fontsize=16)
nx.draw(
    G, pos, node_size=25, node_color=bc_node_colors,
    edge_color='#e0e0e0', width=0.5, with_labels=False
)
if bc_legend_handles:
    plt.legend(handles=bc_legend_handles, loc='best')
plt.axis("equal")
plt.tight_layout()
plt.show()


# --- Plot 2: Visualization by Material (Young's Modulus) ---
# Assign colors based on Young's Modulus (now the 3rd column, index 2)
if data.x.shape[1] == 5:
    modulus_values = data.x[:, 2].numpy()
    # Define colors for matrix (E=5.0) and fiber (E=100.0)
    # Using a small tolerance for float comparison
    material_color_map = {
        5.0: '#8e44ad',    # Purple for Matrix
        100.0: '#f1c40f'  # Yellow for Fiber
    }
    material_node_colors = []
    for E_val in modulus_values:
        if np.isclose(E_val, 5.0):
            material_node_colors.append(material_color_map[5.0])
        elif np.isclose(E_val, 100.0):
            material_node_colors.append(material_color_map[100.0])
        else:
            material_node_colors.append('#bdc3c7') # Default to gray

    # Create legend for materials
    material_legend_handles = [
        plt.Line2D([0], [0], marker='o', color='w', label=f'Matrix (E={5.0})', markersize=10, markerfacecolor=material_color_map.get(5.0)),
        plt.Line2D([0], [0], marker='o', color='w', label=f'Fiber (E={100.0})', markersize=10, markerfacecolor=material_color_map.get(100.0))
    ]
else:
    print("Warning: Node features do not have the expected shape of 5. Cannot plot materials.")
    material_node_colors = '#bdc3c7'
    material_legend_handles = []

# Draw the material definition graph
plt.figure(figsize=(10, 10))
plt.title("Graph Visualization by Material (Young's Modulus)", fontsize=16)
nx.draw(
    G, pos, node_size=25, node_color=material_node_colors,
    edge_color='#e0e0e0', width=0.5, with_labels=False
)
if material_legend_handles:
    plt.legend(handles=material_legend_handles, loc='best')
plt.axis("equal")
plt.tight_layout()
plt.show()
