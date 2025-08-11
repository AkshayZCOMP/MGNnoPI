import os, meshio, torch, numpy as np
from torch_geometric.data import Data
from collections import defaultdict, Counter

def load_msh_to_graph(msh_path):
    """
    Loads a .msh file and converts it into a PyG Data object WITHOUT boundary conditions.
    
    Node features (4): [x, y, E, nu]
    Edge features (4): [dx, dy, dist, angle]
    """
    mesh = meshio.read(msh_path)
    points = mesh.points[:,:2]
    num_nodes = points.shape[0]

    material_properties = {
        1: {'E': 5.0, 'nu': 0.3},
        2: {'E': 100.0, 'nu': 0.3},
        0: {'E': 0.0, 'nu': 0.0}
    }

    #==================================================
    # Step 1: Extract region info from triangles
    #==================================================
    try:
        triangles = mesh.get_cells_type('triangle')
        triangle_regions = mesh.get_cell_data('gmsh:physical', 'triangle')
    except KeyError:
        raise ValueError("No triangle elements or physical region tags found in the mesh.")
    
    node_region_votes = defaultdict(list)
    for tri, region in zip(triangles, triangle_regions):
        for node_index in tri:
            node_region_votes[node_index].append(region)

    region_ids = np.zeros(num_nodes, dtype=int)
    for i in range(num_nodes):
        votes = node_region_votes.get(i)
        if not votes: 
            region_ids[i] = 0
        else:
            region_ids[i] = Counter(votes).most_common(1)[0][0]

    #==================================================
    # Step 2: Build node features tensor (NO BOUNDARY CONDITIONS)
    #==================================================
    # Each node now has 4 features: x, y, E, nu
    node_features = np.zeros((num_nodes, 4))
    node_features[:, 0] = points[:, 0]  # x-coordinates
    node_features[:, 1] = points[:, 1]  # y-coordinates
    
    # Assign material properties based on the region_id of each node
    for i in range(num_nodes):
        region_id = region_ids[i]
        props = material_properties.get(region_id, material_properties[0])
        node_features[i, 2] = props['E']   # Young's Modulus
        node_features[i, 3] = props['nu']  # Poisson's Ratio

    #==================================================
    # Step 3: Extract edges from triangles
    #==================================================
    edge_set = set()
    for tri in triangles:
        edge_set.add(tuple(sorted((tri[0], tri[1]))))
        edge_set.add(tuple(sorted((tri[1], tri[2]))))
        edge_set.add(tuple(sorted((tri[2], tri[0]))))
    
    edge_list = list(edge_set)
    bidirectional_edges = edge_list + [(j,i) for (i,j) in edge_list]
    edge_index = np.array(bidirectional_edges).T

    #==================================================
    # Step 4: Compute edge features 
    #==================================================
    edge_attrs = []
    for i, j in edge_index.T:
        dx = points[j,0] - points[i,0]
        dy = points[j,1] - points[i,1]
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        edge_attrs.append([dx, dy, dist, angle])

    #==================================================
    # Step 5: Convert to PyTorch tensors and package into Data object
    #==================================================
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long) 
    node_tensor = torch.tensor(node_features, dtype=torch.float)
    data = Data(
        x=node_tensor,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr,
        pos=node_tensor[:,:2]
    )
    #==================================================
    # Step 6: Verification step
    #==================================================
    print(f"Graph generated with {data.num_nodes} nodes.")
    print(f"Node feature dimension: {data.num_node_features}") # Should be 4
    print(f"Edge feature dimension: {data.num_edge_features}") # Should be 4

    return data
