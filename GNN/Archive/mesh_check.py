import meshio
import matplotlib.pyplot as plt
import numpy as np
import os

# Path to mesh file
mesh_path = os.path.join("data", "meshes", "horizontal_fibers_rve.msh")

# Load mesh
mesh = meshio.read(mesh_path)

# Extract 2D geometry
points = mesh.points[:, :2]
triangles = mesh.cells_dict.get("triangle", [])
region_tags = mesh.cell_data_dict["gmsh:physical"]["triangle"]

# Create a colormap
unique_regions = np.unique(region_tags)
colors = plt.cm.tab10(np.linspace(0, 1, len(unique_regions)))
region_color_map = {tag: colors[i] for i, tag in enumerate(unique_regions)}

# Plot mesh
fig, ax = plt.subplots(figsize=(6, 6))
for i, tri in enumerate(triangles):
    coords = points[tri]
    region = region_tags[i]
    polygon = plt.Polygon(coords, facecolor=region_color_map[region], edgecolor='black', linewidth=0.2)
    ax.add_patch(polygon)

# Formatting
ax.set_aspect('equal')
ax.set_title("Color-Coded RVE Mesh with Horizontal Fibers")
ax.set_xlabel("x")
ax.set_ylabel("y")
ax.axis('off')
plt.tight_layout()
plt.show()
