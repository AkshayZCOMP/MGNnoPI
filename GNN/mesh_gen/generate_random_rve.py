import gmsh
import os
import random
import numpy as np

def generate_rve_mesh_random(mesh_size, output_filename, L=1.0, fiber_thickness=0.1, min_fibers=1, max_fibers=4):
    """
    Generates a 2D RVE mesh with a random number of horizontal fibers at random vertical positions.
    """
    gmsh.initialize()
    gmsh.model.add(f"rve_random_{random.randint(0, 9999)}")

    # --- 1. Determine random fiber positions ---
    num_fibers = random.randint(min_fibers, max_fibers)
    y_positions = []
    min_gap = 0.05
    y_range = [0, L - fiber_thickness]
    
    while len(y_positions) < num_fibers:
        y_pos = random.uniform(y_range[0], y_range[1])
        is_valid = True
        for existing_y in y_positions:
            if abs(y_pos - existing_y) < (fiber_thickness + min_gap):
                is_valid = False
                break
        if is_valid:
            y_positions.append(y_pos)
    y_positions.sort()

    # --- 2. Create Geometry ---
    matrix_tag = gmsh.model.occ.addRectangle(0, 0, 0, L, L)
    fiber_tags = []
    for y_start in y_positions:
        fiber = gmsh.model.occ.addRectangle(0, y_start, 0, L, fiber_thickness)
        fiber_tags.append((2, fiber))
    gmsh.model.occ.fragment([(2, matrix_tag)], fiber_tags)
    gmsh.model.occ.synchronize()

    # --- 3. Assign Physical Groups for Surfaces ---
    surfaces = gmsh.model.occ.getEntities(dim=2)
    matrix_surfaces, fiber_surfaces = [], []
    for s_dim, s_tag in surfaces:
        com = gmsh.model.occ.getCenterOfMass(s_dim, s_tag)
        is_fiber = False
        for y_start in y_positions:
            if y_start <= com[1] <= (y_start + fiber_thickness):
                fiber_surfaces.append(s_tag)
                is_fiber = True
                break
        if not is_fiber:
            matrix_surfaces.append(s_tag)

    if matrix_surfaces:
        gmsh.model.addPhysicalGroup(2, matrix_surfaces, tag=1)
        gmsh.model.setPhysicalName(2, 1, "Matrix")
    if fiber_surfaces:
        gmsh.model.addPhysicalGroup(2, fiber_surfaces, tag=2)
        gmsh.model.setPhysicalName(2, 2, "Fiber")

    # --- 4. FIX: Use Adjacency Check for Robust Boundary Tagging ---
    all_lines = gmsh.model.occ.getEntities(dim=1)
    left_boundary_lines = []
    right_boundary_lines = []
    tolerance = 1e-5

    for line_dim, line_tag in all_lines:
        bbox = gmsh.model.getBoundingBox(line_dim, line_tag)
        min_x, max_x = bbox[0], bbox[3]
        
        # Check for vertical lines at x=0
        if abs(min_x) < tolerance and abs(max_x) < tolerance:
            # Check that it's an external boundary (adjacent to only 1 surface)
            adj_surfaces, _ = gmsh.model.getAdjacencies(line_dim, line_tag)
            if len(adj_surfaces) == 1:
                 left_boundary_lines.append(line_tag)

        # Check for vertical lines at x=L
        if abs(min_x - L) < tolerance and abs(max_x - L) < tolerance:
            # Check that it's an external boundary
            adj_surfaces, _ = gmsh.model.getAdjacencies(line_dim, line_tag)
            if len(adj_surfaces) == 1:
                right_boundary_lines.append(line_tag)

    if left_boundary_lines:
        gmsh.model.addPhysicalGroup(1, left_boundary_lines, tag=3)
        gmsh.model.setPhysicalName(1, 3, "LeftBoundary_Dirichlet")
    if right_boundary_lines:
        gmsh.model.addPhysicalGroup(1, right_boundary_lines, tag=4)
        gmsh.model.setPhysicalName(1, 4, "RightBoundary_Dirichlet")

    # --- 5. Generate and Save Mesh ---
    gmsh.model.mesh.setSize(gmsh.model.getEntities(0), mesh_size)
    gmsh.model.mesh.generate(2)
    gmsh.write(output_filename)
    gmsh.finalize()
    print(f"Generated mesh with {num_fibers} fibers: {output_filename}")
