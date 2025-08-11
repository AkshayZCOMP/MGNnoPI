import torch
import torch.optim as optim
import os
import matplotlib.pyplot as plt
import sys
from torch_geometric.loader import DataLoader
import random
import glob

# Add the root 'GNN' directory to the Python path by going up three levels
sys.path.append(os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__)))))

from models.pimgn_model import PIMGN
from torch.optim.lr_scheduler import ReduceLROnPlateau

def train_on_diverse_data():
    """
    Trains the PIMGN model on the full diverse dataset, splitting it into
    training and validation sets.
    """
    
    # --- 1. Load All Graph Samples from the Diverse Dataset ---
    base_dir = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
    graph_dir = os.path.join(base_dir, "data", "graphs_diverse")
    
    graph_files = sorted(glob.glob(os.path.join(graph_dir, "graph_sample_*.pt")))
    if not graph_files:
        print(f"Error: No graph samples found in {graph_dir}")
        print("Please run the diverse dataset generation script first.")
        return

    all_data = []
    for graph_path in graph_files:
        try:
            graph = torch.load(graph_path, weights_only=False)
            gt_path = graph_path.replace("graph_sample_", "ground_truth_sample_")
            graph.y = torch.load(gt_path)
            all_data.append(graph)
        except Exception as e:
            print(f"Could not load or process {graph_path}: {e}")
            continue
    
    print(f"Successfully loaded {len(all_data)} graph samples.")

    # --- 2. Split Data into Training and Validation Sets ---
    random.shuffle(all_data)
    split_index = int(len(all_data) * 0.8) # 80% for training, 20% for validation
    train_data = all_data[:split_index]
    val_data = all_data[split_index:]

    print(f"Training on {len(train_data)} samples, validating on {len(val_data)} samples.")

    train_loader = DataLoader(train_data, batch_size=4, shuffle=True)
    val_loader = DataLoader(val_data, batch_size=4, shuffle=False)

    # --- 3. Initialize Model, Optimizer, and Loss Function ---
    if not all_data: return
    node_in_dim = all_data[0].num_node_features
    edge_in_dim = all_data[0].num_edge_features
    hidden_dim = 128
    out_dim = 2
    # --- MODIFIED: Increased model depth ---
    num_layers = 10

    model = PIMGN(node_in_dim, edge_in_dim, hidden_dim, out_dim, num_layers)
    optimizer = optim.Adam(model.parameters(), lr=1e-4)
    scheduler = ReduceLROnPlateau(optimizer, 'min', factor=0.7, patience=50, min_lr=1e-6)
    criterion = torch.nn.MSELoss()
    
    train_losses = []
    val_losses = []

    # --- 4. The Training Loop ---
    print("\nStarting supervised training on diverse dataset...")
    # --- MODIFIED: Increased epochs for the deeper model ---
    num_epochs = 1500
    
    for epoch in range(num_epochs):
        model.train()
        total_train_loss = 0
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
    
    # --- 5. Save the Model and Plot ---
    output_dir = os.path.join(base_dir, "data", "models")
    os.makedirs(output_dir, exist_ok=True)
    model_save_path = os.path.join(output_dir, "supervised_model_diverse.pt")
    torch.save(model.state_dict(), model_save_path)
    print(f"\nTrained model saved to: {model_save_path}")

    plt.figure(figsize=(10, 6))
    plt.plot(train_losses, label='Training Loss')
    plt.plot(val_losses, label='Validation Loss')
    plt.title('Training and Validation Loss on Diverse Dataset')
    plt.xlabel('Epoch')
    plt.ylabel('MSE Loss')
    plt.yscale('log')
    plt.legend()
    plt.grid(True)
    plt.show()

if __name__ == "__main__":
    train_on_diverse_data()
