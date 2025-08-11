import meshio
import matplotlib.pyplot as plt
import numpy as np
import os
import torch
import networkx as nx
from torch_geometric.utils import to_networkx

def combined_visualization(dataset_name="coarse", output_dir="GNN/analysis"):
    """
    Combines the visualization of the RVE mesh and its graph representation,
    and saves the plot to a file.

    Args:
        dataset_name (str): The suffix of the dataset to load (e.g., 'coarse' or 'fine').
        output_dir (str): The directory to save the output plot.
    """
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    mesh_path = os.path.join(base_dir, "data", "meshes", f"horizontal_fibers_rve_{dataset_name}.msh")
    graph_path = os.path.join(base_dir, "data", "graphs", f"horizontal_fibers_graph_{dataset_name}.pt")

    # --- Load Mesh Data ---
    try:
        mesh = meshio.read(mesh_path)
        points_mesh = mesh.points[:, :2]
        triangles_mesh = mesh.cells_dict.get("triangle", [])
        region_tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]
        print(f"Mesh loaded successfully from: {mesh_path}")
    except FileNotFoundError:
        print(f"Error: Mesh file not found at {mesh_path}")
        print("Please ensure you have run 'generate_multiple_datasets.py' first.")
        return
    except Exception as e:
        print(f"An error occurred while loading the mesh: {e}")
        return

    # Create a colormap for mesh regions
    unique_regions = np.unique(region_tags)
    colors_mesh = plt.cm.tab10(np.linspace(0, 1, len(unique_regions)))
    region_color_map = {tag: colors_mesh[i] for i, tag in enumerate(unique_regions)}

    # --- Load Graph Data ---
    try:
        data = torch.load(graph_path, weights_only=False)
        print(f"Graph loaded successfully from: {graph_path}")
    except FileNotFoundError:
        print(f"Error: Graph file not found at {graph_path}")
        print("Please ensure you have run 'generate_multiple_datasets.py' first.")
        return
    except Exception as e:
        print(f"An error occurred while loading the graph: {e}")
        return

    # --- Set up graph properties for visualization ---
    pos_graph = {i: data.pos[i].tolist() for i in range(data.num_nodes)}
    G_graph = to_networkx(data, to_undirected=True)

    # --- Initialize Plot ---
    fig, axes = plt.subplots(1, 3, figsize=(24, 8))
    fig.suptitle(f"Combined Visualization of RVE ({dataset_name.capitalize()} Mesh)", fontsize=18)

    # --- Plot 1: Mesh Visualization ---
    ax_mesh = axes[0]
    for i, tri in enumerate(triangles_mesh):
        coords = points_mesh[tri]
        region = region_tags[i]
        polygon = plt.Polygon(coords, facecolor=region_color_map[region], edgecolor='black', linewidth=0.2)
        ax_mesh.add_patch(polygon)

    ax_mesh.set_aspect('equal')
    ax_mesh.set_title("RVE Mesh (Material Regions)", fontsize=14)
    ax_mesh.set_xlabel("x")
    ax_mesh.set_ylabel("y")
    ax_mesh.axis('off')

    # Create legend for mesh regions
    mesh_legend_handles = [
        plt.Line2D([0], [0], marker='s', color='w', label='Matrix (Region 1)', markersize=10, markerfacecolor=region_color_map.get(1)),
        plt.Line2D([0], [0], marker='s', color='w', label='Fiber (Region 2)', markersize=10, markerfacecolor=region_color_map.get(2))
    ]
    ax_mesh.legend(handles=mesh_legend_handles, loc='best')

    # --- Plot 2: Graph Visualization by Boundary Condition (One-Hot Encoded) ---
    ax_bc = axes[1]
    # Check for 7 features: [x, y, E, nu, is_interior, is_left_bc, is_right_bc]
    if data.x.shape[1] == 7:
        # Extract the one-hot encoded part (last 3 columns)
        bc_one_hot = data.x[:, 4:].numpy()
        # Find the index of the '1' in each row to get the BC ID
        bc_ids = np.argmax(bc_one_hot, axis=1)

        # Map BC IDs to colors
        # 0: interior, 1: left_bc, 2: right_bc
        bc_color_map = {0: '#3498db', 1: '#e67e22', 2: '#2ecc71'} # Blue, Orange, Green
        bc_node_colors = [bc_color_map.get(bc_id, '#bdc3c7') for bc_id in bc_ids]
        
        nx.draw(
            G_graph, pos_graph, node_size=25, node_color=bc_node_colors,
            edge_color='#e0e0e0', width=0.5, with_labels=False, ax=ax_bc
        )
        ax_bc.set_title("Graph (Boundary Conditions)", fontsize=14)
        ax_bc.set_aspect('equal')
        ax_bc.axis("off")

        bc_legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', label='Interior (BC=[1,0,0])', markersize=10, markerfacecolor=bc_color_map.get(0)),
            plt.Line2D([0], [0], marker='o', color='w', label='Fixed (BC=[0,1,0])', markersize=10, markerfacecolor=bc_color_map.get(1)),
            plt.Line2D([0], [0], marker='o', color='w', label='Strained (BC=[0,0,1])', markersize=10, markerfacecolor=bc_color_map.get(2))
        ]
        ax_bc.legend(handles=bc_legend_handles, loc='best')
    else:
        ax_bc.set_title("Graph (BC) - N/A (Node features not one-hot)", fontsize=14)


    # --- Plot 3: Graph Visualization by Material (Young's Modulus) ---
    ax_material = axes[2]
    # Check for at least 3 features so we can access Young's Modulus
    if data.x.shape[1] >= 3:
        modulus_values = data.x[:, 2].numpy()
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

        nx.draw(
            G_graph, pos_graph, node_size=25, node_color=material_node_colors,
            edge_color='#e0e0e0', width=0.5, with_labels=False, ax=ax_material
        )
        ax_material.set_title("Graph (Material Properties)", fontsize=14)
        ax_material.set_aspect('equal')
        ax_material.axis("off")

        material_legend_handles = [
            plt.Line2D([0], [0], marker='o', color='w', label=f'Matrix (E={5.0})', markersize=10, markerfacecolor=material_color_map.get(5.0)),
            plt.Line2D([0], [0], marker='o', color='w', label=f'Fiber (E={100.0})', markersize=10, markerfacecolor=material_color_map.get(100.0))
        ]
        ax_material.legend(handles=material_legend_handles, loc='best')
    else:
        ax_material.set_title("Graph (Material) - N/A (Node features unexpected)", fontsize=14)

    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    # --- Save the Figure ---
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, f"combined_visualization_{dataset_name}.png")
    plt.savefig(output_path)
    plt.close(fig) # Close the figure to free up memory
    print(f"Visualization for {dataset_name} saved to: {output_path}")


if __name__ == "__main__":
    # Ensure the main analysis directory exists
    output_directory = os.path.join(os.path.dirname(os.path.dirname(os.path.abspath(__file__))), "GNN", "analysis")
    os.makedirs(output_directory, exist_ok=True)
    
    # Visualize both 'coarse' and 'fine' datasets
    combined_visualization(dataset_name="coarse", output_dir=output_directory)
    combined_visualization(dataset_name="fine", output_dir=output_directory)