import os
import sys
import torch

# Add necessary paths to import modules
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'mesh_gen')))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'graph_builder')))

# Import functions from your existing scripts
from mesh_gen.generate_rve import generate_rve_mesh
from graph_builder.generategroundtruth import generate_displacement_field
#from graph_builder.mesh_to_graph import load_msh_to_graph
from graph_builder.Mesh_to_graph_onehot import load_msh_to_graph

def generate_datasets():
    """
    Generates multiple mesh, ground truth, and graph datasets with varying mesh sizes.
    """
    dataset_configs = [
        {'name': 'coarse', 'mesh_size': 0.05},  # Original size
        {'name': 'fine', 'mesh_size': 0.02},    # Smaller mesh size
    ]

    base_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(os.path.join(base_data_dir, 'meshes'), exist_ok=True)
    os.makedirs(os.path.join(base_data_dir, 'graphs'), exist_ok=True)

    for config in dataset_configs:
        name = config['name']
        mesh_size = config['mesh_size']
        
        print(f"\n--- Generating dataset: {name} (Mesh Size: {mesh_size}) ---")

        # 1. Generate Mesh
        mesh_filename = f"horizontal_fibers_rve_{name}.msh"
        generate_rve_mesh(mesh_size=mesh_size, output_filename=mesh_filename)
        mesh_path = os.path.join(base_data_dir, 'meshes', mesh_filename)

        # 2. Generate Ground Truth Displacements
        displacement_filename = f"ground_truth_displacements_{name}.pt"
        displacement_output_path = os.path.join(base_data_dir, 'graphs', displacement_filename)
        generate_displacement_field(mesh_path, displacement_output_path)

        # 3. Convert Mesh to Graph
        graph_filename = f"horizontal_fibers_graph_{name}.pt"
        graph_output_path = os.path.join(base_data_dir, 'graphs', graph_filename)
        graph = load_msh_to_graph(mesh_path)
        torch.save(graph, graph_output_path)
        print(f"Graph saved successfully to: {graph_output_path}")

    print("\nAll datasets generated successfully!")

if __name__ == "__main__":
    generate_datasets()