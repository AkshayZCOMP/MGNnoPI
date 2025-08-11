import gmsh
import os
import numpy as np
import argparse

def generate_rve_mesh(mesh_size, output_filename, L=1.0, fiber_thickness=1.0/9.0, gap=(1.0 - 2 * (1.0/9.0)) / 3.0, tolerance=1e-5):
    """
    Generates a 2D RVE mesh with horizontal fibers and assigns physical groups for regions
    and boundaries.

    Args:
        mesh_size (float): The characteristic length for mesh elements. Smaller values result in finer meshes.
        output_filename (str): The name of the output .msh file.
        L (float): Side length of the unit square RVE.
        fiber_thickness (float): Thickness of the horizontal fibers.
        gap (float): Vertical gap between the fibers.
        tolerance (float): Tolerance for geometric comparisons.
    """
    gmsh.initialize()
    gmsh.model.add("rve_with_bc")

    # --- Create geometry ---
    matrix_tag = gmsh.model.occ.addRectangle(0, 0, 0, L, L)

    fiber_tags = []
    for i in range(2):
        y_start = gap * (i + 1) + fiber_thickness * i
        fiber = gmsh.model.occ.addRectangle(0, y_start, 0, L, fiber_thickness)
        fiber_tags.append((2, fiber))

    gmsh.model.occ.fragment([(2, matrix_tag)], fiber_tags)
    gmsh.model.occ.synchronize()

    # --- Assign Physical Groups for Surfaces (Fibers/Matrix) ---
    surfaces = gmsh.model.occ.getEntities(dim=2)

    def get_surface_centroid_y(tag):
        com = gmsh.model.occ.getCenterOfMass(2, tag[1])
        return com[1]

    sorted_surfaces = sorted(surfaces, key=get_surface_centroid_y)

    if len(sorted_surfaces) == 5:
        matrix_surfaces = [sorted_surfaces[0][1], sorted_surfaces[2][1], sorted_surfaces[4][1]]
        gmsh.model.addPhysicalGroup(2, matrix_surfaces, tag=1)
        gmsh.model.setPhysicalName(2, 1, "Matrix")

        fiber_surfaces = [sorted_surfaces[1][1], sorted_surfaces[3][1]]
        gmsh.model.addPhysicalGroup(2, fiber_surfaces, tag=2)
        gmsh.model.setPhysicalName(2, 2, "Fiber")
    else:
        print(f"Warning: Expected 5 surfaces after fragmentation, but found {len(sorted_surfaces)}. Physical groups may be incorrect.")

    # --- Assign Physical Groups for Boundaries (Lines) ---
    gmsh.model.occ.synchronize()
    lines = gmsh.model.occ.getEntities(dim=1)

    left_boundary_lines = []
    right_boundary_lines = []

    for line_dim, line_tag in lines:
        bbox = gmsh.model.getBoundingBox(line_dim, line_tag)
        min_x, max_x = bbox[0], bbox[3]
        
        if abs(min_x) < tolerance and abs(max_x) < tolerance:
            left_boundary_lines.append(line_tag)
            
        if abs(min_x - L) < tolerance and abs(max_x - L) < tolerance:
            right_boundary_lines.append(line_tag)

    if left_boundary_lines:
        gmsh.model.addPhysicalGroup(1, left_boundary_lines, tag=3)
        gmsh.model.setPhysicalName(1, 3, "LeftBoundary_Dirichlet")

    if right_boundary_lines:
        gmsh.model.addPhysicalGroup(1, right_boundary_lines, tag=4)
        gmsh.model.setPhysicalName(1, 4, "RightBoundary_Dirichlet")

    # --- Generate and Save Mesh ---
    all_points = gmsh.model.occ.getEntities(dim=0)
    for p_dim, p_tag in all_points:
        gmsh.model.mesh.setSize([(p_dim, p_tag)], mesh_size)

    gmsh.model.mesh.generate(2)

    output_dir = os.path.join("data", "meshes")
    os.makedirs(output_dir, exist_ok=True)
    output_path = os.path.join(output_dir, output_filename)
    gmsh.write(output_path)
    gmsh.finalize()

    print(f"Mesh with boundary conditions successfully saved to: {output_path}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='Generate RVE mesh with specified parameters.')
    parser.add_argument('--mesh_size', type=float, default=0.05, help='Characteristic mesh size.')
    parser.add_argument('--output_filename', type=str, default="horizontal_fibers_rve.msh", help='Output filename for the mesh.')
    parser.add_argument('--L', type=float, default=1.0, help='Unit square side length.')
    parser.add_argument('--fiber_thickness', type=float, default=1.0/9.0, help='Thickness of the fibers.')
    parser.add_argument('--gap', type=float, default=(1.0 - 2 * (1.0/9.0)) / 3.0, help='Gap between fibers.')
    parser.add_argument('--tolerance', type=float, default=1e-5, help='Tolerance for coordinate comparison.')

    args = parser.parse_args()

    generate_rve_mesh(args.mesh_size, args.output_filename, args.L, args.fiber_thickness, args.gap, args.tolerance)