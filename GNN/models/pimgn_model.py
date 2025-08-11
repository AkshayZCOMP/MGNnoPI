import torch
import torch.nn as nn
from torch_geometric.nn import MessagePassing
import torch.nn.functional as F


#===============================================
# 1. Node Encoder
#===============================================
class NodeEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(NodeEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )
    def forward(self,x):
        return self.mlp(x)

#===============================================
# 2. Edge Encoder
#===============================================
class EdgeEncoder(nn.Module):
    def __init__(self, in_dim, hidden_dim):
        super(EdgeEncoder, self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
        )

    def forward(self, edge_attr):
        return self.mlp(edge_attr)
    
#===============================================
# 3. Message Passing Layer with LayerNorm
#===============================================
class GNNLayer(MessagePassing):
    def __init__(self, hidden_dim):
        super(GNNLayer,self).__init__(aggr='add')
        
        # MLP to update node embeddings after messages are received
        self.mlp = nn.Sequential(
            nn.Linear(2 * hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim)
        )
        
        # --- ADDED: Layer Normalization ---
        # LayerNorm is applied to the node features to stabilize training.
        self.layer_norm = nn.LayerNorm(hidden_dim)

    def forward(self,x,edge_index,edge_attr):
        # This is the entry point that triggers the message passing.
        # The output of propagate() will be passed to the update() method.
        return self.propagate(edge_index, x=x, edge_attr=edge_attr)
    
    def message(self, x_j, edge_attr):
        # x_j: features of neighboring nodes [num_edges, hidden_dim]
        # edge_attr: features of connecting edges [num_edges, hidden_dim]
        # The message is a simple sum of the neighbor's features and the edge's features.
        return x_j + edge_attr
    
    def update(self,aggr_out,x):
        # aggr_out: The result of message aggregation [num_nodes, hidden_dim]
        # x: Original node embeddings [num_nodes, hidden_dim]
        
        # --- ADDED: Apply LayerNorm ---
        # We normalize the aggregated messages before further processing.
        aggr_out = self.layer_norm(aggr_out)
        
        # Concatenate the node's current state (x) and the normalized incoming messages (aggr_out)
        new_embedding_input = torch.cat([x, aggr_out], dim=1)
        
        # Pass the concatenated vector through the MLP to get the update vector
        update_vector = self.mlp(new_embedding_input)
        
        # Use a residual connection: add the update to the original embedding
        return x + update_vector
    
#===============================================
# 4. Decoder MLP
#===============================================
class Decoder(nn.Module):
    def __init__(self, hidden_dim, out_dim):
        super(Decoder,self).__init__()
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim,out_dim)
        )

    def forward(self, x):
        return self.mlp(x)

#===============================================
# 5. PIMGN Model - Putting it all together
#===============================================
class PIMGN(nn.Module):
    def __init__(self, node_in_dim, edge_in_dim, hidden_dim, out_dim, num_layers=3):
        super(PIMGN, self).__init__()
        
        self.node_encoder = NodeEncoder(node_in_dim,hidden_dim)
        self.edge_encoder = EdgeEncoder(edge_in_dim, hidden_dim)

        # Create a stack of GNN layers for the processor
        self.processor = nn.ModuleList()
        for _ in range(num_layers):
            self.processor.append(GNNLayer(hidden_dim))
            
        # Final decoder to get the physical prediction
        self.decoder = Decoder(hidden_dim, out_dim)

    def forward(self, data):
        # Unpack data object
        x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

        # 1. ENCODE
        x = self.node_encoder(x)
        edge_attr = self.edge_encoder(edge_attr)

        # 2. PROCESS
        for layer in self.processor:
            x = layer(x, edge_index, edge_attr)
            
        # 3. DECODE
        return self.decoder(x)



# import torch
# import torch.nn as nn
# from torch_geometric.nn import MessagePassing
# import torch.nn.functional as F


# #===============================================
# # 1. Node Encoder   input ftr > hidden dim usising rlu for non linearity in . hidden using lienar
# #===============================================
# class NodeEncoder(nn.Module):#nn.module is pytorch compoent inherent standard funcitonaly from pytorch base module
#     # The __init__ method is the constructor. It sets up the layers of the module.
#     def __init__(self, in_dim, hidden_dim):# super calls the iinit method of the parent class nn.Module
#         super(NodeEncoder, self).__init__()
#         # self.mlp holds our sequence of operations. nn.Sequential is a container
#         # that will pass data through each of the listed layers in order.
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),

#         )
#     # The 'forward' method defines what happens when data is passed through.
#     def forward(self,x):
#         return self.mlp(x)

# #===============================================
# #edge encoder with fwd
# #===============================================
# class EdgeEncoder(nn.Module): #nn.module is pytorch compoent inherent standard funcitonaly from pytorch base module
#     """Encodes edge features into a higher-dimensional latent space."""
#     # This class is identical in function to the NodeEncoder but acts on edge features.
#     def __init__(self, in_dim, hidden_dim):
#         super(EdgeEncoder, self).__init__()# super calls the iinit method of the parent class nn.Module
#         self.mlp = nn.Sequential(
#             nn.Linear(in_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim),
#         )

#     def forward(self, edge_attr):
#         return self.mlp(edge_attr)
    
# #===============================================
# # Message Passing Layer
# #===============================================
# class GNNLayer(MessagePassing):
#     "single later of message passing core of GNN"
#     def __init__(self, hidden_dim):
#         super(GNNLayer,self).__init__(aggr='add')  # 'add' aggregation
        
#         #mlp to udpate node empeddings after messages recieved
#         # *** FIX ***: The input dimension for this MLP is 2 * hidden_dim because we concatenate
#         # the node's own features and the aggregated messages.
#         self.mlp = nn.Sequential(
#             nn.Linear(2 * hidden_dim, hidden_dim),
#             nn.ReLU(),
#             nn.Linear(hidden_dim, hidden_dim)
#         )

#     def forward(self,x,edge_index,edge_attr):
#         #start propragation procss
#         # This is the entry point that triggers the message passing.
#         return self.propagate(edge_index,x=x,edge_attr=edge_attr)
    
#     def message(self, x_j, edge_attr):
#         #x_j = featrues of neighborind nodes [e, hidden dim]
#         #defines message sent from neighbor node j to central node i 
#         # edge attr: ftr of connecting edge [e, hidden_dim]
#         # The message is a simple sum, combining info about the neighbor and the connection.
#         return x_j + edge_attr
    
#     def update(self,aggr_out,x):
#         #defines how to update central nodes embedding (x) using the aggregated messages
#         # aggr_out: the result of message aggregation [N, hidden_dim]
#         # x: original node embeddings [N, hidden_dim]
        
#         #concatenate curent node state x and incoming messages aggr_out into single VECTOR
#         new_embedding = self.mlp(torch.cat([x, aggr_out],dim=1))
        
#         # passes concat through mlp defined above, and results new node embedding that dpenonds on current state and neighbnors
#         # A "residual connection" is used (x + ...), which helps stabilize training.
#         return x + new_embedding
    
# #===============================================
# #decoder mlp
# #===============================================
# class Decoder(nn.Module):
#     '''decodes final node embeddings into output featuer (2d displacement)'''
#     def __init__(self, hidden_dim, out_dim):
#         super(Decoder,self).__init__()
#         self.mlp = nn.Sequential(
#             nn.Linear(hidden_dim, hidden_dim),
#             nn.ReLU(),
#             # The final layer maps from the hidden dimension to the desired output dimension.
#             nn.Linear(hidden_dim,out_dim)
#         )

#     def forward(self, x):
#         return self.mlp(x)

# #===============================================
# # PImgN Model
# #===============================================
# # *** FIX ***: Inherit from nn.Module (capital M)
# class PIMGN(nn.Module):
#     '''complete model that connnects all the pieces'''
#     def __init__(self, node_in_dim, edge_in_dim, hidden_dim, out_dim, num_layers=3):
#         super(PIMGN, self).__init__()
        
#         # --- Create instances of all the model components ---
#         self.node_encoder = NodeEncoder(node_in_dim,hidden_dim)
#         self.edge_encoder = EdgeEncoder(edge_in_dim, hidden_dim)

#         #create stack of GNN layers for processor
#         self.processor = nn.ModuleList()
#         for _ in range(num_layers):
#             self.processor.append(GNNLayer(hidden_dim))
            
#         #final decoder to get physical pred
#         self.decoder = Decoder(hidden_dim, out_dim)

#     def forward(self, data):
#         '''define forawrd pass of the model'''
        
#         # *** FIX ***: Unpack data object correctly.
#         x, edge_index, edge_attr = data.x, data.edge_index, data.edge_attr

#         # --- 1. ENCODE ---
#         # The encoders must be called at the start of the forward pass.
#         x = self.node_encoder(x)
#         edge_attr = self.edge_encoder(edge_attr)

#         # --- 2. PROCESS ---
#         # Pass the encoded data through the message passing layers.
#         for layer in self.processor:
#             x = layer(x, edge_index, edge_attr)
            
#         # --- 3. DECODE ---
#         # Use the decoder to get the final prediction from the processed node features.
#         return self.decoder(x)
