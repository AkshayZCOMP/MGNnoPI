import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.tri as mtri

# --- Add Project Root to Python Path ---
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pimgn_model import PIMGN
from torch_geometric.data import Data

def plot_deformation(ax, node_pos, triangles, displacements, title, magnification=5.0, cmap='viridis'):
    """
    Helper function to plot the deformed mesh with displacement magnitude contours.

    Args:
        ax (matplotlib.axes.Axes): The matplotlib axes object to plot on.
        node_pos (np.array): The original positions of the nodes [num_nodes, 2].
        triangles (np.array): The triangle connectivity [num_triangles, 3].
        displacements (np.array): The displacement vector for each node [num_nodes, 2].
        title (str): The title for the plot.
        magnification (float): A factor to visually exaggerate the deformation.
        cmap (str): The colormap to use for the plot.
    """
    deformed_pos = node_pos + displacements * magnification
    
    tri_obj = mtri.Triangulation(deformed_pos[:, 0], deformed_pos[:, 1], triangles)
    
    displacement_magnitude = np.sqrt(displacements[:, 0]**2 + displacements[:, 1]**2)
    
    tpc = ax.tripcolor(tri_obj, displacement_magnitude, shading='gouraud', cmap=cmap)
    
    plt.colorbar(tpc, ax=ax, label='Displacement Magnitude')
    
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X-coordinate')
    ax.set_ylabel('Y-coordinate')
    ax.set_aspect('equal', 'box')
    ax.grid(True, linestyle='--', alpha=0.6)


def compare_deformations(dataset_name="coarse"):
    """
    Loads the ground truth and the GNN's prediction for a specified dataset,
    then plots them side-by-side along with a plot showing the error between them.

    Args:
        dataset_name (str): The suffix of the dataset to load (e.g., 'coarse' or 'fine').
    """
    # --- 1. Define Paths and Check for Files ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    
    graph_path = os.path.join(base_dir, "data", "graphs", f"horizontal_fibers_graph_{dataset_name}.pt")
    model_path = os.path.join(base_dir, "data", "models", "supervised_model_multidata.pt") # Updated model path
    ground_truth_path = os.path.join(base_dir, "data", "graphs", f"ground_truth_displacements_{dataset_name}.pt")
    mesh_path = os.path.join(base_dir, "data", "meshes", f"horizontal_fibers_rve_{dataset_name}.msh")

    if not all(os.path.exists(p) for p in [graph_path, model_path, ground_truth_path, mesh_path]):
        print("Error: Required files not found for the specified dataset or model.")
        print("Please ensure you have run 'generate_multiple_datasets.py' and 'supervisedtrain.py' first.")
        return

    # --- 2. Load Data ---
    print(f"Loading data for {dataset_name} dataset...")
    graph = torch.load(graph_path, weights_only=False)
    true_displacements = torch.load(ground_truth_path).numpy()
    
    original_positions = graph.pos.numpy()
    import meshio
    mesh = meshio.read(mesh_path)
    triangles = mesh.get_cells_type("triangle")

    # --- 3. Load Trained Model and Predict ---
    print("Loading trained model and making prediction...")
    model = PIMGN(
        node_in_dim=graph.num_node_features,
        edge_in_dim=graph.num_edge_features,
        hidden_dim=128,
        out_dim=2,
        num_layers=5
    )
    model.load_state_dict(torch.load(model_path))
    model.eval()

    with torch.no_grad():
        predicted_displacements = model(graph).numpy()

    # --- 4. Create Side-by-Side Plots with Error ---
    print("Generating plots...")
    fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
    
    plot_deformation(ax1, original_positions, triangles, true_displacements, f"Ground Truth Deformation (FEM) - {dataset_name.capitalize()}")
    
    plot_deformation(ax2, original_positions, triangles, predicted_displacements, f"GNN Predicted Deformation - {dataset_name.capitalize()}")

    error_displacements = true_displacements - predicted_displacements
    error_magnitude = np.sqrt(np.sum(error_displacements**2, axis=1))

    tri_obj_orig = mtri.Triangulation(original_positions[:, 0], original_positions[:, 1], triangles)
    
    tpc_error = ax3.tripcolor(tri_obj_orig, error_magnitude, shading='gouraud', cmap='inferno')
    
    plt.colorbar(tpc_error, ax=ax3, label='Error Magnitude')
    ax3.set_title(f"Displacement Error (Difference) - {dataset_name.capitalize()}", fontsize=14)
    ax3.set_xlabel('X-coordinate')
    ax3.set_ylabel('Y-coordinate')
    ax3.set_aspect('equal', 'box')
    ax3.grid(True, linestyle='--', alpha=0.6)
    
    fig.suptitle(f'Visual Comparison of Deformation Fields and Error for {dataset_name.capitalize()} Dataset', fontsize=18)
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    plt.show()

if __name__ == "__main__":
    os.makedirs("GNN/analysis", exist_ok=True)
    # You can change 'coarse' to 'fine' to visualize the deformation for the other dataset
    compare_deformations(dataset_name="fine")