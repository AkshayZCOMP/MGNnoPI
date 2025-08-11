import os, meshio, torch, numpy as np
from torch_geometric.data import Data
from collections import defaultdict, Counter

def load_msh_to_graph(msh_path):
    """
    Loads a .msh file with boundary conditions and converts it into a PyG Data object.
    
    Node features will now be: [x, y, Youngs_Modulus, Poissons_Ratio, bc_id]
    """
    mesh = meshio.read(msh_path)
    points = mesh.points[:,:2]
    num_nodes = points.shape[0]

    # --- Define Material Properties ---
    # This dictionary maps the physical group IDs from Gmsh to material constants.
    material_properties = {
        1: {'E': 5.0, 'nu': 0.3},    # Region ID 1: Matrix
        2: {'E': 100.0, 'nu': 0.3}, # Region ID 2: Fiber
        0: {'E': 0.0, 'nu': 0.0}     # Default for nodes not in a region
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
            # gets most common from list of votes [1,2,2] that goes to format of [2:2, 1:1] 0,0 extracts that item and assigns to node i
            region_ids[i] = Counter(votes).most_common(1)[0][0]

    #==================================================
    #step2 boundary condioton info
    #==================================================
    bc_ids = np.zeros(num_nodes, dtype=int)
    try:
        #get all 1d line element defintiosn (which 2 nodes make up each line) 
        lines = mesh.get_cells_type('line')
        # get physical tags for lines
        line_tags = mesh.get_cell_data('gmsh:physical', 'line')
        for line, tag in zip(lines, line_tags):
            for node_index in line:
                # if left bc set 1
                if tag == 3:
                    bc_ids[node_index] = 1 
                elif tag == 4:
                    bc_ids[node_index] = 2 
    except KeyError: 
        print("No boundary condition tags found for lines. Assuming all nodes are interior (bc_id=0).")

    #==================================================
    #step 3 extract edges from triangles
    #==================================================
    edge_set = set()
    for tri in triangles:
        #sort so (b,a) = (a,b)
        edge_set.add(tuple(sorted((tri[0], tri[1]))))
        edge_set.add(tuple(sorted((tri[1], tri[2]))))
        edge_set.add(tuple(sorted((tri[2], tri[0]))))
    
    edge_list = list(edge_set)
    # creates directed edges by adding reverse of each edge to list
    bidirectional_edges = edge_list + [(j,i) for (i,j) in edge_list]
    #transpose to get shape [2, num_edges] required format for PyG
    edge_index = np.array(bidirectional_edges).T

    #==================================================
    # step 4 build node features tensor
    #==================================================
    # each node now has 5 features: x, y, Youngs_Modulus, Poissons_Ratio, bc_id
    node_features = np.zeros((num_nodes, 5))
    node_features[:, 0] = points[:, 0]  # x-coordinates 1 col
    node_features[:, 1] = points[:, 1]  # y-coordinates 2 col
    
    # Assign material properties based on the region_id of each node
    for i in range(num_nodes):
        region_id = region_ids[i]
        props = material_properties.get(region_id, material_properties[0])
        node_features[i, 2] = props['E']   # Young's Modulus 3rd col
        node_features[i, 3] = props['nu']  # Poisson's Ratio 4th col

    node_features[:, 4] = bc_ids  # bc_id 5th col

    #==================================================
    # step 5 compute edge features
    #==================================================
    edge_attrs = []
    # iterate through pairs of (source, target) nodes
    for i, j in edge_index.T:
        dx = points[j,0] - points[i,0] # x comp of position vector from i to j
        dy = points[j,1] - points[i,1] # y comp of position vector from i to j
        dist = np.sqrt(dx**2 + dy**2) # euclidean distance
        angle = np.arctan2(dy, dx) # angle of edge
        # flag for crossing material boundaries
        material_jump = 1.0 if region_ids[i] != region_ids[j] else 0.0
        # append all
        edge_attrs.append([dx, dy, dist, angle, material_jump])

    #==================================================
    # step 6 convert to PyTorch tensors and package into Data object
    #==================================================
    edge_attr = torch.tensor(edge_attrs, dtype=torch.float)
    edge_index_tensor = torch.tensor(edge_index, dtype=torch.long) 
    node_tensor = torch.tensor(node_features, dtype=torch.float)
    data = Data(
        x=node_tensor,
        edge_index=edge_index_tensor,
        edge_attr=edge_attr,
        # Store original positions separately for easy access
        pos=node_tensor[:,:2]
    )
    #==================================================
    # step 7 verification step
    #==================================================
    # Prints the total number of nodes in the generated graph.
    print(f"Graph generated with {data.num_nodes} nodes.")
    # Prints the number of features per node (should be 5).
    print(f"Node feature dimension: {data.num_node_features}")
    # Prints the count of nodes that were tagged as part of the fixed left boundary.
    print(f"Dirichlet_fixed (BC=1) nodes found: {np.sum(bc_ids == 1)}")
    # Prints the count of nodes that were tagged as part of the strained right boundary.
    print(f"Dirichlet_strained (BC=2) nodes found: {np.sum(bc_ids == 2)}")
    # Prints the count of nodes that were not on a tagged boundary (the interior nodes).
    print(f"Interior (BC=0) nodes found: {np.sum(bc_ids == 0)}")

    # Returns the completed graph object.
    return data

if __name__ == "__main__":
    input_path = os.path.join("data", "meshes", "horizontal_fibers_rve.msh")
    output_dir = os.path.join("data", "graphs")
    output_path = os.path.join(output_dir, "horizontal_fibers_graph.pt")
    os.makedirs(output_dir, exist_ok=True)
    
    print(f"Loading mesh from: {input_path}")
    graph = load_msh_to_graph(input_path)
    
    print(f"\nGraph successfully generated and saved to: {output_path}")
    # Verify the new shape of the node features
    print(f"Sample node features (first node): {graph.x[0]}")
    torch.save(graph, output_path)
    print(f"Graph saved successfully to: {output_path}")

