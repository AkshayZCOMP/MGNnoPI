import meshio
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import networkx as nx
from torch_geometric.utils import to_networkx
import glob
import sys

# --- MODIFIED: Adjust system path to find modules from a subfolder ---
# This adds the root 'GNN' directory to the path.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

def visualize_all_samples(output_dir_name="graphvis"):
    """
    Loads every sample from the diverse dataset and creates a combined PNG visualization
    for each one, saving it to a specified output directory.
    """
    # --- MODIFIED: Adjust base directory path to work from a subfolder ---
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    mesh_dir = os.path.join(base_dir, "data", "meshes_diverse")
    graph_dir = os.path.join(base_dir, "data", "graphs_diverse")
    output_dir = os.path.join(base_dir, output_dir_name)

    # --- 1. Find all available graph samples ---
    graph_files = sorted(glob.glob(os.path.join(graph_dir, "graph_sample_*.pt")))
    if not graph_files:
        print(f"Error: No graph samples found in {graph_dir}")
        print("Please ensure you have run 'generate_diverse_dataset.py' first.")
        return

    os.makedirs(output_dir, exist_ok=True)
    print(f"Saving all visualizations to: {output_dir}")

    # --- 2. Loop through every sample ---
    for graph_path in graph_files:
        try:
            sample_index = int(os.path.basename(graph_path).split('_')[-1].split('.')[0])
        except (IndexError, ValueError):
            print(f"Could not parse sample index from filename: {graph_path}")
            continue
        
        mesh_path = os.path.join(mesh_dir, f"rve_sample_{sample_index}.msh")
        print(f"Processing sample #{sample_index}...")

        # --- Load Data ---
        try:
            mesh = meshio.read(mesh_path)
            points_mesh = mesh.points[:, :2]
            triangles_mesh = mesh.cells_dict.get("triangle", [])
            region_tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]
            data = torch.load(graph_path, weights_only=False)
        except Exception as e:
            print(f"Skipping sample {sample_index} due to a loading error: {e}")
            continue

        # --- Prepare for Plotting ---
        pos_graph = {i: data.pos[i].tolist() for i in range(data.num_nodes)}
        G_graph = to_networkx(data, to_undirected=True)

        fig, axes = plt.subplots(1, 3, figsize=(24, 8))
        fig.suptitle(f"Diverse Dataset Visualization (Sample #{sample_index})", fontsize=18)

        # Plot 1: Mesh Visualization
        ax_mesh = axes[0]
        region_color_map = {1: '#8e44ad', 2: '#f1c40f'}
        for i, tri in enumerate(triangles_mesh):
            coords = points_mesh[tri]
            region = region_tags[i]
            polygon = plt.Polygon(coords, facecolor=region_color_map.get(region, '#bdc3c7'), edgecolor='black', linewidth=0.2)
            ax_mesh.add_patch(polygon)
        ax_mesh.set_aspect('equal')
        ax_mesh.set_title("RVE Mesh", fontsize=14)
        ax_mesh.axis('off')

        # Plot 2: Graph by Boundary Condition
        ax_bc = axes[1]
        if data.x.shape[1] == 7:
            bc_one_hot = data.x[:, 4:].numpy()
            bc_ids = np.argmax(bc_one_hot, axis=1)
            bc_color_map = {0: '#3498db', 1: '#e67e22', 2: '#2ecc71'}
            bc_node_colors = [bc_color_map.get(bc_id, '#bdc3c7') for bc_id in bc_ids]
            nx.draw(G_graph, pos_graph, node_size=15, node_color=bc_node_colors, edge_color='#e0e0e0', width=0.5, with_labels=False, ax=ax_bc)
        else:
            nx.draw(G_graph, pos_graph, node_size=15, edge_color='#e0e0e0', width=0.5, with_labels=False, ax=ax_bc)
        ax_bc.set_title("Graph by BC", fontsize=14)
        ax_bc.set_aspect('equal')
        ax_bc.axis("off")

        # Plot 3: Graph by Material
        ax_material = axes[2]
        modulus_values = data.x[:, 2].numpy()
        material_color_map = {5.0: '#8e44ad', 100.0: '#f1c40f'}
        material_node_colors = [material_color_map.get(E_val, '#bdc3c7') for E_val in np.round(modulus_values)]
        nx.draw(G_graph, pos_graph, node_size=15, node_color=material_node_colors, edge_color='#e0e0e0', width=0.5, with_labels=False, ax=ax_material)
        ax_material.set_title("Graph by Material", fontsize=14)
        ax_material.set_aspect('equal')
        ax_material.axis("off")

        plt.tight_layout(rect=[0, 0.03, 1, 0.95])
        
        # --- Save the Combined Figure ---
        output_path = os.path.join(output_dir, f"sample_visualization_{sample_index}.png")
        plt.savefig(output_path)
        plt.close(fig)

    print("\nFinished generating all visualizations.")

if __name__ == "__main__":
    visualize_all_samples()
