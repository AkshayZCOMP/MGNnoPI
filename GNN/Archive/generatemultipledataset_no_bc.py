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
# --- IMPORT FROM THE NEW SCRIPT ---
from graph_builder.mesh_to_graph_no_bc import load_msh_to_graph

def generate_datasets_no_bc():
    """
    Generates multiple mesh, ground truth, and graph datasets with varying mesh sizes,
    WITHOUT including boundary conditions in the graph features.
    """
    dataset_configs = [
        {'name': 'coarse', 'mesh_size': 0.05},
        {'name': 'fine', 'mesh_size': 0.02},
    ]

    base_data_dir = os.path.join(os.path.dirname(__file__), '..', 'data')
    os.makedirs(os.path.join(base_data_dir, 'meshes'), exist_ok=True)
    os.makedirs(os.path.join(base_data_dir, 'graphs'), exist_ok=True)

    for config in dataset_configs:
        name = config['name']
        mesh_size = config['mesh_size']
        
        print(f"\n--- Generating dataset: {name} (Mesh Size: {mesh_size}) ---")

        # 1. Generate Mesh (this can be reused)
        mesh_filename = f"horizontal_fibers_rve_{name}.msh"
        mesh_path = os.path.join(base_data_dir, 'meshes', mesh_filename)
        if not os.path.exists(mesh_path):
             generate_rve_mesh(mesh_size=mesh_size, output_filename=mesh_filename)
       
        # 2. Generate Ground Truth Displacements (this can be reused)
        displacement_filename = f"ground_truth_displacements_{name}.pt"
        displacement_output_path = os.path.join(base_data_dir, 'graphs', displacement_filename)
        if not os.path.exists(displacement_output_path):
            generate_displacement_field(mesh_path, displacement_output_path)

        # 3. Convert Mesh to Graph using the "no_bc" script
        print(f"\n--- Converting mesh to graph with NO boundary conditions for {name} ---")
        # --- SAVE WITH A NEW FILENAME ---
        graph_filename_no_bc = f"horizontal_fibers_graph_{name}_no_bc.pt"
        graph_output_path = os.path.join(base_data_dir, 'graphs', graph_filename_no_bc)
        
        graph = load_msh_to_graph(mesh_path) # This calls the "no_bc" version
        torch.save(graph, graph_output_path)
        print(f"Graph with no BC features saved successfully to: {graph_output_path}")

    print("\nAll datasets (no BC) generated successfully!")

if __name__ == "__main__":
    generate_datasets_no_bc()
