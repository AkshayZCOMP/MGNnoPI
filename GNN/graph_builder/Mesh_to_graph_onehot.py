import os, meshio, torch, numpy as np
from torch_geometric.data import Data
from collections import defaultdict, Counter

def load_msh_to_graph(msh_path):
    """
    Loads a .msh file and converts it into a PyG Data object.
    
    Node features (7): [x, y, E, nu, is_interior, is_left_bc, is_right_bc]
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
    #step 1 extract region info from triangles det if node is matrix or fiber
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
    #step 2 build node features tensor with ONE-HOT for BCs
    #==================================================
    # each node now has 7 features: x, y, E, nu, is_interior, is_left_bc, is_right_bc
    node_features = np.zeros((num_nodes, 7))
    node_features[:, 0] = points[:, 0]  # x-coordinates 1 col
    node_features[:, 1] = points[:, 1]  # y-coordinates 2 col
    
    # Assign material properties based on the region_id of each node
    for i in range(num_nodes):
        region_id = region_ids[i]
        props = material_properties.get(region_id, material_properties[0])
        node_features[i, 2] = props['E']   # Young's Modulus 3rd col
        node_features[i, 3] = props['nu']  # Poisson's Ratio 4th col

    # Default all nodes to interior [1, 0, 0] for the one-hot columns
    node_features[:, 4] = 1.0 # is_interior column

    try:
        lines = mesh.get_cells_type('line')
        line_tags = mesh.get_cell_data('gmsh:physical', 'line')
        for line, tag in zip(lines, line_tags):
            for node_index in line:
                if tag == 3: # Left boundary
                    # Set to left_bc: [0, 1, 0]
                    node_features[node_index, 4] = 0.0 # Not interior
                    node_features[node_index, 5] = 1.0 # Is left_bc
                elif tag == 4: # Right boundary
                    # Set to right_bc: [0, 0, 1]
                    node_features[node_index, 4] = 0.0 # Not interior
                    node_features[node_index, 6] = 1.0 # Is right_bc
    except KeyError: 
        print("No boundary condition tags found for lines. All nodes are assumed to be interior.")

    #==================================================
    #step 3 extract edges from triangles
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
    # step 4 compute edge features 
    #==================================================
    edge_attrs = []
    for i, j in edge_index.T:
        dx = points[j,0] - points[i,0]
        dy = points[j,1] - points[i,1]
        dist = np.sqrt(dx**2 + dy**2)
        angle = np.arctan2(dy, dx)
        # The 'material_jump' feature has been removed
        edge_attrs.append([dx, dy, dist, angle])

    #==================================================
    # step 5 convert to PyTorch tensors and package into Data object
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
    # step 6 verification step
    #==================================================
    print(f"Graph generated with {data.num_nodes} nodes.")
    print(f"Node feature dimension: {data.num_node_features}") # Should be 7
    print(f"Edge feature dimension: {data.num_edge_features}") # Should now be 4
    print(f"Interior nodes found (BC=[1,0,0]): {np.sum(node_features[:, 4] == 1)}")
    print(f"Left BC nodes found (BC=[0,1,0]): {np.sum(node_features[:, 5] == 1)}")
    print(f"Right BC nodes found (BC=[0,0,1]): {np.sum(node_features[:, 6] == 1)}")

    return data

if __name__ == "__main__":
    input_path = os.path.join("..", "data", "meshes", "horizontal_fibers_rve.msh")
    output_dir = os.path.join("..", "data", "graphs")
    output_path = os.path.join(output_dir, "horizontal_fibers_graph.pt")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading mesh from: {input_path}")
    graph = load_msh_to_graph(input_path)
    
    print(f"\nGraph successfully generated and saved to: {output_path}")
    print(f"Sample node features (first node): {graph.x[0]}")
    print(f"Sample edge features (first edge): {graph.edge_attr[0]}")
    torch.save(graph, output_path)
    print("Graph file overwritten with new features.")