import torch
import numpy as np
import os
import sys
import matplotlib.pyplot as plt
import matplotlib.tri as mtri
from matplotlib.colors import ListedColormap
import random
import glob
import meshio
import csv

# Add the root 'GNN' directory to the Python path by going up three levels
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.pimgn_model import PIMGN

def calculate_effective_modulus(points, triangles, region_ids, displacements, strain_load=0.01):
    """
    Calculates the effective Young's modulus in the x-direction from a displacement field.
    This is done by volume-averaging the stress over all elements.
    """
    E_matrix, nu_matrix = 5.0, 0.3
    E_fiber, nu_fiber = 100.0, 0.3

    def get_C(E, nu):
        return (E / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])

    C_matrix, C_fiber = get_C(E_matrix, nu_matrix), get_C(E_fiber, nu_fiber)
    total_area, volume_averaged_stress = 0, np.zeros(3)
    U = displacements.flatten()

    for i, tri in enumerate(triangles):
        C = C_matrix if region_ids[i] == 1 else C_fiber
        p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
        area = 0.5 * np.linalg.det(np.array([[p2[0]-p1[0], p3[0]-p1[0]], [p2[1]-p1[1], p3[1]-p1[1]]]))
        if abs(area) < 1e-12: continue
        total_area += area
        y23, y31, y12 = p2[1]-p3[1], p3[1]-p1[1], p1[1]-p2[1]
        x32, x13, x21 = p3[0]-p2[0], p1[0]-p3[0], p2[0]-p1[0]
        B = (1/(2*area)) * np.array([[y23,0,y31,0,y12,0], [0,x32,0,x13,0,x21], [x32,y23,x13,y31,x21,y12]])
        element_dofs = np.ravel([[2*node, 2*node+1] for node in tri])
        u_element = U[element_dofs]
        element_strain = B @ u_element
        element_stress = C @ element_strain
        volume_averaged_stress += element_stress * area

    average_stress = volume_averaged_stress / total_area if total_area > 0 else np.zeros(3)
    E_eff_x = average_stress[0] / strain_load if strain_load != 0 else 0
    return E_eff_x

def plot_deformed_materials(ax, original_positions, triangles, region_ids, displacements, title, magnification=5.0):
    """ Helper function to plot the deformed mesh, coloring elements by material region. """
    deformed_pos = original_positions + displacements * magnification
    color_matrix, color_fiber = '#8e44ad', '#f1c40f'
    cmap = ListedColormap([color_matrix, color_fiber])
    tri_obj = mtri.Triangulation(deformed_pos[:, 0], deformed_pos[:, 1], triangles)
    ax.tripcolor(tri_obj, (region_ids - 1), shading='flat', cmap=cmap, edgecolor='black', linewidth=0.2)
    legend_handles = [plt.Line2D([0],[0], marker='s', color='w', label=l, markersize=10, markerfacecolor=c) for l,c in [('Matrix', color_matrix), ('Fiber', color_fiber)]]
    ax.legend(handles=legend_handles, loc='best', fontsize=10)
    ax.set_title(title, fontsize=14)
    ax.set_xlabel('X-coordinate'); ax.set_ylabel('Y-coordinate')
    ax.set_aspect('equal', 'box'); ax.grid(True, linestyle='--', alpha=0.6)

def visualize_all_predictions():
    """
    Loads the trained model, iterates through ALL samples, plots the deformation comparison,
    and saves all results to a CSV file.
    """
    # --- 1. Define Paths ---
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    model_path = os.path.join(base_dir, "data", "models", "supervised_model_diverse.pt")
    graph_dir = os.path.join(base_dir, "data", "graphs_diverse")
    mesh_dir = os.path.join(base_dir, "data", "meshes_diverse")
    output_dir = os.path.join(base_dir, "analysisdiverse")
    os.makedirs(output_dir, exist_ok=True)

    # --- 2. Find all Samples ---
    graph_files = sorted(glob.glob(os.path.join(graph_dir, "graph_sample_*.pt")))
    if not graph_files:
        print(f"Error: No graph samples found in {graph_dir}"); return
        
    print(f"Found {len(graph_files)} samples to process. Visualizations will be saved to: {output_dir}")
    
    # --- 3. Prepare for results collection ---
    results_data = []
    model = None # Load model only once

    # --- 4. Loop through ALL samples ---
    for graph_path in graph_files:
        sample_index = int(os.path.basename(graph_path).split('_')[-1].split('.')[0])
        mesh_path = os.path.join(mesh_dir, f"rve_sample_{sample_index}.msh")
        print(f"\nProcessing sample #{sample_index}...")

        # Load Data for the Sample
        try:
            graph = torch.load(graph_path, weights_only=False)
            true_displacements = torch.load(graph_path.replace("graph_sample_", "ground_truth_sample_")).numpy()
            mesh = meshio.read(mesh_path)
            triangles = mesh.get_cells_type("triangle")
            region_ids = mesh.get_cell_data("gmsh:physical", "triangle")
            original_positions = graph.pos.numpy()
        except Exception as e:
            print(f"  > ERROR: Could not load data for sample {sample_index}. Skipping. ({e})")
            continue

        # Load Model (only on first loop) and Make Prediction
        if model is None:
            # --- IMPORTANT: Initialize model with the same architecture as training ---
            model = PIMGN(graph.num_node_features, graph.num_edge_features, 128, 2, num_layers=10)
            try:
                model.load_state_dict(torch.load(model_path))
                model.eval()
                print("Model loaded successfully.")
            except FileNotFoundError:
                print(f"FATAL: Model file not found at {model_path}. Aborting."); return
        
        with torch.no_grad():
            predicted_displacements = model(graph).numpy()

        # Calculate Effective Moduli
        E_eff_fem = calculate_effective_modulus(original_positions, triangles, region_ids, true_displacements)
        E_eff_mgn = calculate_effective_modulus(original_positions, triangles, region_ids, predicted_displacements)
        percent_error = 100 * abs(E_eff_fem - E_eff_mgn) / E_eff_fem if E_eff_fem != 0 else 0
        results_data.append([sample_index, E_eff_fem, E_eff_mgn, percent_error])
        print(f"  > FEM Modulus: {E_eff_fem:.4f} | GNN Modulus: {E_eff_mgn:.4f} ({percent_error:.2f}% Error)")

        # Create and Save Plots
        fig, (ax1, ax2, ax3) = plt.subplots(1, 3, figsize=(24, 7))
        plot_deformed_materials(ax1, original_positions, triangles, region_ids, true_displacements, f"Ground Truth Def. (Sample #{sample_index})")
        plot_deformed_materials(ax2, original_positions, triangles, region_ids, predicted_displacements, f"GNN Predicted Def. (Sample #{sample_index})")
        
        error_magnitude = np.sqrt(np.sum((true_displacements - predicted_displacements)**2, axis=1))
        tri_obj_orig = mtri.Triangulation(original_positions[:, 0], original_positions[:, 1], triangles)
        tpc_error = ax3.tripcolor(tri_obj_orig, error_magnitude, shading='gouraud', cmap='inferno')
        plt.colorbar(tpc_error, ax=ax3, label='Error Magnitude')
        ax3.set_title(f"Displacement Error (Difference)", fontsize=14)
        ax3.set_xlabel('X-coordinate'); ax3.set_ylabel('Y-coordinate')
        ax3.set_aspect('equal', 'box'); ax3.grid(True, linestyle='--', alpha=0.6)
        
        fig.suptitle(f'Visual Comparison (Sample #{sample_index})\n'
                     f'FEM Modulus: {E_eff_fem:.2f} | GNN Modulus: {E_eff_mgn:.2f} ({percent_error:.2f}% Error)',
                     fontsize=16)
        plt.tight_layout(rect=[0, 0.03, 1, 0.92])
        
        output_path = os.path.join(output_dir, f"diverse_prediction_comparison_{sample_index}.png")
        plt.savefig(output_path)
        plt.close(fig)

    # --- 5. Save all results to a CSV file ---
    csv_path = os.path.join(output_dir, "modulus_comparison.csv")
    with open(csv_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(["Sample_ID", "FEM_Modulus", "GNN_Modulus", "Percent_Error"])
        writer.writerows(results_data)
    print(f"\nAll results saved to {csv_path}")

if __name__ == "__main__":
    visualize_all_predictions()
