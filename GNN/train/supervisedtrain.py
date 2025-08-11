import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import sys
from torch_geometric.loader import DataLoader # Import the DataLoader
import random

# Add the parent directory (GNN) to the Python path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from models.pimgn_model import PIMGN
from torch.optim.lr_scheduler import ReduceLROnPlateau

def run_supervised_training():
    """
    Trains the PIMGN model in a supervised manner using a DataLoader
    to handle multiple graph datasets simultaneously.
    """
    
    # --- 1. Load All Graph Datasets ---
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    graph_dir = os.path.join(base_dir, "data", "graphs")
    
    dataset_configs = ['coarse', 'fine']
    all_data = []

    for name in dataset_configs:
        #graph_path = os.path.join(graph_dir, f"horizontal_fibers_graph_{name}_no_bc.pt")
        graph_path = os.path.join(graph_dir, f"horizontal_fibers_graph_{name}.pt")
        gt_path = os.path.join(graph_dir, f"ground_truth_displacements_{name}.pt")
        
        if not os.path.exists(graph_path) or not os.path.exists(gt_path):
            print(f"Error: Data for '{name}' not found. Please run data generation.")
            return
            
        graph = torch.load(graph_path, weights_only=False)
        # Attach ground truth displacements to the 'y' attribute of the graph
        graph.y = torch.load(gt_path)
        all_data.append(graph)
        print(f"Loaded dataset: {name} (Nodes: {graph.num_nodes})")

    # For this example, we'll use both for training and validation
    # In a real scenario, you'd have a separate, larger test set.
    train_loader = DataLoader(all_data, batch_size=2, shuffle=True)
    val_loader = DataLoader(all_data, batch_size=2, shuffle=False)

    # --- 2. Initialize Model, Optimizer, and Loss Function ---
    node_in_dim = all_data[0].num_node_features
    edge_in_dim = all_data[0].num_edge_features
    hidden_dim = 128
    out_dim = 2
    num_layers = 5

    model = PIMGN(node_in_dim, edge_in_dim, hidden_dim, out_dim, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=50, min_lr=1e-6)
    criterion = torch.nn.MSELoss()
    
    train_losses = []
    val_losses = []

    # --- 3. The Training Loop ---
    print("\nStarting supervised training with DataLoader...")
    num_epochs = 1000
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
        # The loader creates batches of graphs (here, a batch of 2)
        for batch in train_loader:
            optimizer.zero_grad()
            predicted_y = model(batch)
            loss = criterion(predicted_y, batch.y)
            loss.backward()
            torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
            optimizer.step()
            total_train_loss += loss.item()
        
        avg_train_loss = total_train_loss / len(train_loader)
        train_losses.append(avg_train_loss)

        # --- Validation Step ---
        model.eval()
        total_val_loss = 0
        with torch.no_grad():
            for batch in val_loader:
                predicted_y = model(batch)
                val_loss = criterion(predicted_y, batch.y)
                total_val_loss += val_loss.item()

        avg_val_loss = total_val_loss / len(val_loader)
        val_losses.append(avg_val_loss)
        scheduler.step(avg_val_loss)

        if epoch % 50 == 0:
            print(f"Epoch {epoch:04d} | Train Loss: {avg_train_loss:.8f} | Val Loss: {avg_val_loss:.8f}")

    print("\nSupervised training finished.")
    
    # --- 4. Save the Model and Plot ---
    output_dir = os.path.join(base_dir, "data", "models")
    os.makedirs(output_dir, exist_ok=True)
    
    
    #model_save_path = os.path.join(output_dir, "supervised_model_no_bc.pt")
    model_save_path = os.path.join(output_dir, "supervised_model_multidata.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTrained model saved to: {model_save_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss Curves')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    run_supervised_training()