import meshio
import numpy as np
import torch
import os
from scipy.sparse import lil_matrix
from scipy.sparse.linalg import spsolve

def generate_displacement_field(mesh_path, displacement_output_path):
    """
    A dedicated script that:
    1. Solves a 2D FEM problem for a given mesh and material properties.
    2. Saves the resulting full nodal displacement field [ux, uy] as the ground truth.
    """
    
    # --- 1. Load Mesh and Define Parameters ---
    print(f"Loading mesh from {mesh_path}...")
    mesh = meshio.read(mesh_path)
    points = mesh.points[:, :2]
    triangles = mesh.get_cells_type("triangle")
    
    # Use the new material properties
    E_matrix, nu_matrix = 5.0, 0.3
    E_fiber, nu_fiber = 100.0, 0.3
    strain_load = 0.01

    # --- 2. Assemble Global Stiffness Matrix K ---
    print("Assembling global stiffness matrix K...")
    num_nodes = points.shape[0]
    K = lil_matrix((2 * num_nodes, 2 * num_nodes))
    region_ids = mesh.get_cell_data("gmsh:physical", "triangle")

    def get_C(E, nu):
        return (E / (1 - nu**2)) * np.array([[1, nu, 0], [nu, 1, 0], [0, 0, (1-nu)/2]])
    C_matrix, C_fiber = get_C(E_matrix, nu_matrix), get_C(E_fiber, nu_fiber)

    for i, tri in enumerate(triangles):
        C = C_matrix if region_ids[i] == 1 else C_fiber
        p1, p2, p3 = points[tri[0]], points[tri[1]], points[tri[2]]
        
        area = 0.5 * np.linalg.det(np.array([[p2[0]-p1[0], p3[0]-p1[0]], [p2[1]-p1[1], p3[1]-p1[1]]]))
        
        if abs(area) < 1e-12: continue
        
        y23, y31, y12 = p2[1] - p3[1], p3[1] - p1[1], p1[1] - p2[1]
        x32, x13, x21 = p3[0] - p2[0], p1[0] - p3[0], p2[0] - p1[0]

        B = (1/(2*area)) * np.array([
            [y23,0,y31,0,y12,0],
            [0,x32,0,x13,0,x21],
            [x32,y23,x13,y31,x21,y12]
        ])
        
        Ke = B.T @ C @ B * abs(area)

        for r in range(3):
            for c in range(3):
                dofs = [2*tri[r], 2*tri[r]+1, 2*tri[c], 2*tri[c]+1]
                K[np.ix_([2*tri[r], 2*tri[r]+1], [2*tri[c], 2*tri[c]+1])] += Ke[np.ix_([2*r, 2*r+1], [2*c, 2*c+1])]

    # --- 3. Apply Boundary Conditions and Solve for Displacements ---
    print("Solving for nodal displacements...")
    lines = mesh.get_cells_type("line")
    line_tags = mesh.get_cell_data("gmsh:physical", "line")
    left_boundary_nodes = set(lines[line_tags == 3].flatten())
    right_boundary_nodes = set(lines[line_tags == 4].flatten())

    U = np.zeros(2 * num_nodes)
    fixed_dofs = []
    
    for node_idx in left_boundary_nodes:
        fixed_dofs.extend([2 * node_idx, 2 * node_idx + 1])
    
    for node_idx in right_boundary_nodes:
        fixed_dofs.append(2 * node_idx)
        U[2 * node_idx] = strain_load * np.max(points[:, 0])

    active_dofs = np.setdiff1d(np.arange(2 * num_nodes), fixed_dofs)
    K_reduced = K[np.ix_(active_dofs, active_dofs)].tocsr()
    F_reduced = np.zeros(len(active_dofs)) - K[np.ix_(active_dofs, fixed_dofs)] @ U[fixed_dofs]
    U[active_dofs] = spsolve(K_reduced, F_reduced)
    
    # --- 4. Save the Displacement Field as the Ground Truth ---
    # Reshape the flat U vector ([2*num_nodes]) into a 2D tensor ([num_nodes, 2])
    displacement_tensor = torch.tensor(U.reshape(-1, 2), dtype=torch.float)
    
    # Save the tensor to the specified path
    torch.save(displacement_tensor, displacement_output_path)
    
    print("\n--- Process Complete ---")
    print(f"Solved for {len(U)//2} nodal displacements.")
    print(f"Ground truth displacement field saved successfully to: {displacement_output_path}")


if __name__ == "__main__":
    # Ensure the output directories exist
    os.makedirs("GNN/analysis", exist_ok=True)
    os.makedirs("data/graphs", exist_ok=True)
    
    # Define input and output paths
    mesh_input_path = os.path.join("data", "meshes", "horizontal_fibers_rve.msh")
    displacement_output_pt = os.path.join("data", "graphs", "ground_truth_displacements.pt")
    
    # Run the function
    generate_displacement_field(mesh_input_path, displacement_output_pt)