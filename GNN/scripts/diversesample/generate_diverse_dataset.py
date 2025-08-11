import os
import sys
import torch
import random

# --- MODIFIED: Adjust system path to find modules from a subfolder ---
# This adds the root 'GNN' directory to the path, allowing imports to work correctly.
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', '..')))

# Import the necessary functions
from mesh_gen.generate_random_rve import generate_rve_mesh_random
from graph_builder.generategroundtruth import generate_displacement_field
from graph_builder.Mesh_to_graph_onehot import load_msh_to_graph 

def create_diverse_dataset(num_samples=50, mesh_size=0.05):
    """
    Generates a diverse dataset of RVEs with random fiber layouts.
    
    Args:
        num_samples (int): The total number of unique RVE samples to generate.
        mesh_size (float): The mesh size to use for all samples.
    """
    # --- MODIFIED: Adjust base directory path to find the 'data' folder ---
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    base_data_dir = os.path.join(base_dir, 'data')
    mesh_dir = os.path.join(base_data_dir, 'meshes_diverse')
    graph_dir = os.path.join(base_data_dir, 'graphs_diverse')
    
    os.makedirs(mesh_dir, exist_ok=True)
    os.makedirs(graph_dir, exist_ok=True)

    for i in range(num_samples):
        print(f"\n--- Generating Sample {i+1}/{num_samples} ---")
        
        # --- Define unique filenames for this sample ---
        mesh_filename = os.path.join(mesh_dir, f"rve_sample_{i}.msh")
        gt_filename = os.path.join(graph_dir, f"ground_truth_sample_{i}.pt")
        graph_filename = os.path.join(graph_dir, f"graph_sample_{i}.pt")

        # 1. Generate a new random mesh
        generate_rve_mesh_random(
            mesh_size=mesh_size,
            output_filename=mesh_filename,
            min_fibers=1,
            max_fibers=4
        )

        # 2. Generate the ground truth displacement field for it
        generate_displacement_field(mesh_filename, gt_filename)

        # 3. Convert the new mesh to a graph
        graph = load_msh_to_graph(mesh_filename)
        torch.save(graph, graph_filename)
        print(f"Graph saved successfully to: {graph_filename}")

    print(f"\nSuccessfully generated a diverse dataset with {num_samples} samples.")

if __name__ == "__main__":
    # This will create a dataset of 10 samples for a quick test.
    # You can increase this number for the full training.
    create_diverse_dataset(num_samples=100)
