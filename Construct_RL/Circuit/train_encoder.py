# train_encoder.py
import os
import random
import re
from collections import defaultdict

import torch
import torch.nn as nn
from torch.nn import TripletMarginLoss
from torch.optim import Adam
import torch.nn.functional as F

from torch_geometric.nn import GATv2Conv, global_mean_pool
from torch_geometric.data import Dataset, Data
from torch_geometric.loader import DataLoader
import numpy as np
from tqdm import tqdm

# --- Configuration ---
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
NUM_CIRCUITS = 100000
TEST_SET_SIZE = 10000
VAL_SET_SIZE = 9000
BATCH_SIZE = 64
EPOCHS = 20
LEARNING_RATE = 1e-4
EMBEDDING_DIM = 128

# --- Helper Function to Parse Netlists ---

COMP_VOCAB = {'V': 0, 'R': 1, 'C': 2, 'L': 3, 'E': 4, 'I': 5, 'G':6, 'F':7, 'H':8}
NUM_COMP_TYPES = len(COMP_VOCAB)

def parse_netlist_to_graph(file_path):
    with open(file_path, 'r') as f:
        lines = f.readlines()
    nodes, edges, edge_attrs = set([0]), [], []
    pattern = re.compile(r"([a-zA-Z]+)\d*\s+(\d+)\s+(\d+)\s+([0-9eE.+-]+)")
    for line in lines:
        match = pattern.match(line)
        if not match: continue
        comp_str, n1_str, n2_str, val_str = match.groups()
        comp_type = COMP_VOCAB.get(comp_str[0].upper(), -1)
        if comp_type == -1: continue
        n1, n2, value = int(n1_str), int(n2_str), float(val_str)
        nodes.update([n1, n2])
        edges.extend([(n1, n2), (n2, n1)])
        type_vec = F.one_hot(torch.tensor(comp_type), num_classes=NUM_COMP_TYPES).float()
        val_vec = torch.tensor([np.log10(value + 1e-12)]).float()
        attr = torch.cat([type_vec, val_vec])
        edge_attrs.extend([attr, attr])
    node_map = {node_id: i for i, node_id in enumerate(sorted(list(nodes)))}
    num_nodes = len(nodes)
    edge_index = torch.tensor([[node_map[u], node_map[v]] for u, v in edges], dtype=torch.long).t().contiguous()
    node_features = torch.ones((num_nodes, 1), dtype=torch.float)
    edge_features = torch.stack(edge_attrs)
    return Data(x=node_features, edge_index=edge_index, edge_attr=edge_features)

# --- Custom PyG Dataset for Triplets ---

class CircuitTripletDataset(Dataset):
    def __init__(self, root_dir, indices):
        super().__init__(root_dir)
        self.indices = indices
        self.indices_set = set(indices)
        self.circuits_dir = os.path.join(root_dir, 'circuits')
        
        self.closest_map = self._read_relationship_file(os.path.join(root_dir, 'closest.txt'))
        self.furthest_map = self._read_relationship_file(os.path.join(root_dir, 'furthest.txt'))
        
    def _read_relationship_file(self, file_path):
        mapping = defaultdict(list)
        with open(file_path, 'r') as f:
            for line in f:
                parts = [p for p in line.strip().split(',') if p]
                if not parts: continue
                anchor_id = int(parts[0])
                related_ids = [int(x) for x in parts[1:]]
                mapping[anchor_id] = related_ids
        return mapping

    def __len__(self):
        return len(self.indices)

    # --- THIS IS THE CORRECTED METHOD NAME ---
    def __getitem__(self, idx):
        # The idx is an integer from 0 to len(self)-1.
        # We use it to get the specific circuit index for our split.
        anchor_idx = self.indices[idx]
        
        valid_positives = [p for p in self.closest_map[anchor_idx] if p in self.indices_set]
        valid_negatives = [n for n in self.furthest_map[anchor_idx] if n in self.indices_set]

        if not valid_positives: positive_idx = random.choice(self.indices)
        else: positive_idx = random.choice(valid_positives)
        
        if not valid_negatives: negative_idx = random.choice(self.indices)
        else: negative_idx = random.choice(valid_negatives)
        
        while positive_idx == anchor_idx: positive_idx = random.choice(self.indices)
        while negative_idx == anchor_idx: negative_idx = random.choice(self.indices)

        anchor_graph = parse_netlist_to_graph(os.path.join(self.circuits_dir, f"circuit_{anchor_idx}.txt"))
        positive_graph = parse_netlist_to_graph(os.path.join(self.circuits_dir, f"circuit_{positive_idx}.txt"))
        negative_graph = parse_netlist_to_graph(os.path.join(self.circuits_dir, f"circuit_{negative_idx}.txt"))

        return anchor_graph, positive_graph, negative_graph

# --- GNN Encoder Model ---

class CircuitGNNEncoder(torch.nn.Module):
    def __init__(self, input_node_dim, edge_feature_dim, hidden_dim, output_dim, num_heads=4):
        super().__init__()
        self.conv1 = GATv2Conv(input_node_dim, hidden_dim, heads=num_heads, edge_dim=edge_feature_dim)
        self.conv2 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=num_heads, edge_dim=edge_feature_dim)
        self.conv3 = GATv2Conv(hidden_dim * num_heads, hidden_dim, heads=1, edge_dim=edge_feature_dim)
        self.mlp = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim * 2), nn.ReLU(),
            nn.Linear(hidden_dim * 2, output_dim)
        )

    def forward(self, data):
        x, edge_index, edge_attr, batch = data.x, data.edge_index, data.edge_attr, data.batch
        x = F.elu(self.conv1(x, edge_index, edge_attr))
        x = F.elu(self.conv2(x, edge_index, edge_attr))
        x = self.conv3(x, edge_index, edge_attr)
        graph_embedding = global_mean_pool(x, batch)
        output = self.mlp(graph_embedding)
        return output

# --- Training and Evaluation Functions ---

def train_epoch(model, loader, optimizer, loss_fn):
    model.train()
    total_loss = 0
    for anchor, positive, negative in tqdm(loader, desc="Training"):
        anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
        optimizer.zero_grad()
        anchor_emb, positive_emb, negative_emb = model(anchor), model(positive), model(negative)
        loss = loss_fn(anchor_emb, positive_emb, negative_emb)
        loss.backward()
        optimizer.step()
        total_loss += loss.item()
    return total_loss / len(loader)

def evaluate(model, loader, loss_fn):
    model.eval()
    total_loss, correct, total = 0, 0, 0
    with torch.no_grad():
        for anchor, positive, negative in tqdm(loader, desc="Evaluating"):
            anchor, positive, negative = anchor.to(DEVICE), positive.to(DEVICE), negative.to(DEVICE)
            anchor_emb, positive_emb, negative_emb = model(anchor), model(positive), model(negative)
            loss = loss_fn(anchor_emb, positive_emb, negative_emb)
            total_loss += loss.item()
            dist_pos = F.pairwise_distance(anchor_emb, positive_emb)
            dist_neg = F.pairwise_distance(anchor_emb, negative_emb)
            correct += torch.sum(dist_pos < dist_neg).item()
            total += anchor_emb.size(0)
    return total_loss / len(loader), correct / total

# --- Main Execution ---
if __name__ == '__main__':
    indices = np.random.permutation(NUM_CIRCUITS)
    test_indices, val_indices, train_indices = indices[:TEST_SET_SIZE], indices[TEST_SET_SIZE:TEST_SET_SIZE + VAL_SET_SIZE], indices[TEST_SET_SIZE + VAL_SET_SIZE:]
    print(f"Dataset split: {len(train_indices)} train, {len(val_indices)} validation, {len(test_indices)} test.")

    train_dataset = CircuitTripletDataset(root_dir='.', indices=train_indices.tolist())
    val_dataset = CircuitTripletDataset(root_dir='.', indices=val_indices.tolist())
    test_dataset = CircuitTripletDataset(root_dir='.', indices=test_indices.tolist())

    # --- DEBUGGING CHANGE: Set num_workers=0 ---
    # Once working, you can change this back to 4 or more for speed.
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True, num_workers=0)
    val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False, num_workers=0)

    EDGE_FEAT_DIM = NUM_COMP_TYPES + 1
    model = CircuitGNNEncoder(
        input_node_dim=1, edge_feature_dim=EDGE_FEAT_DIM,
        hidden_dim=256, output_dim=EMBEDDING_DIM
    ).to(DEVICE)
    loss_fn = TripletMarginLoss(margin=1.0)
    optimizer = Adam(model.parameters(), lr=LEARNING_RATE)

    print(f"Model initialized on {DEVICE}. Total parameters: {sum(p.numel() for p in model.parameters())}")
    best_val_acc = 0
    for epoch in range(EPOCHS):
        train_loss = train_epoch(model, train_loader, optimizer, loss_fn)
        val_loss, val_acc = evaluate(model, val_loader, loss_fn)
        print(f"Epoch {epoch+1}/{EPOCHS} | Train Loss: {train_loss:.4f} | Val Loss: {val_loss:.4f} | Val Acc: {val_acc:.4f}")
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), 'best_encoder.pth')
            print(f"  -> New best model saved with validation accuracy: {best_val_acc:.4f}")

    print("\n--- Final Test Evaluation ---")
    model.load_state_dict(torch.load('best_encoder.pth'))
    test_loss, test_acc = evaluate(model, test_loader, loss_fn)
    print(f"Test Loss: {test_loss:.4f} | Test Accuracy: {test_acc:.4f}")