import sys
import argparse
import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid, Amazon
from ogb.nodeproppred import PygNodePropPredDataset
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.utils import to_undirected
import numpy as np

from stealgnn import GraphGenerator, SurrogateModel, TypeIAttack, TypeIIAttack, TypeIIIAttack, evaluate_models

def create_masks(num_nodes, train_ratio=0.6, val_ratio=0.2):
    indices = np.random.permutation(num_nodes)
    train_size = int(num_nodes * train_ratio)
    val_size = int(num_nodes * val_ratio)
    
    train_mask = torch.zeros(num_nodes, dtype=torch.bool)
    val_mask = torch.zeros(num_nodes, dtype=torch.bool)
    test_mask = torch.zeros(num_nodes, dtype=torch.bool)
    
    train_mask[indices[:train_size]] = True
    val_mask[indices[train_size:train_size+val_size]] = True
    test_mask[indices[train_size+val_size:]] = True
    
    return train_mask, val_mask, test_mask

def load_dataset_and_create_victim_model(dataset_name, device):
    if dataset_name == 'cora':
        dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
        data = dataset[0].to(device)
    elif dataset_name == 'computers':
        dataset = Amazon(root='/tmp/Amazon', name='Computers', transform=NormalizeFeatures())
        data = dataset[0].to(device)
        data.edge_index = to_undirected(data.edge_index)
        train_mask, val_mask, test_mask = create_masks(data.num_nodes)
        data.train_mask, data.val_mask, data.test_mask = train_mask.to(device), val_mask.to(device), test_mask.to(device)
    elif dataset_name == 'pubmed':
        dataset = Planetoid(root='/tmp/Pubmed', name='Pubmed', transform=NormalizeFeatures())
        data = dataset[0].to(device)
    elif dataset_name == 'ogb-arxiv':
        dataset = PygNodePropPredDataset(name='ogbn-arxiv', transform=NormalizeFeatures())
        data = dataset[0].to(device)
        split_idx = dataset.get_idx_split()
        data.train_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.val_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.test_mask = torch.zeros(data.num_nodes, dtype=torch.bool)
        data.train_mask[split_idx['train']] = True
        data.val_mask[split_idx['valid']] = True
        data.test_mask[split_idx['test']] = True
        data.train_mask, data.val_mask, data.test_mask = data.train_mask.to(device), data.val_mask.to(device), data.test_mask.to(device)
    else:
        raise ValueError("Invalid dataset name. Choose 'cora', 'computers', 'pubmed', or 'ogb-arxiv'.")

    input_dim, hidden_dim, output_dim = data.num_features, 16, dataset.num_classes
    victim_model = SurrogateModel(input_dim, hidden_dim, output_dim).to(device)
    return dataset, data, victim_model

def train_victim_model(victim_model, data, dataset_name, epochs=200, lr=0.01, weight_decay=5e-4):
    optimizer = optim.Adam(victim_model.parameters(), lr=lr, weight_decay=weight_decay)
    criterion = nn.NLLLoss()
    
    for epoch in range(epochs):
        victim_model.train()
        optimizer.zero_grad()
        out = victim_model(data.x, data.edge_index)
        loss = criterion(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            victim_model.eval()
            with torch.no_grad():
                val_out = victim_model(data.x, data.edge_index)
                val_loss = criterion(val_out[data.val_mask], data.y[data.val_mask])
                val_acc = (val_out[data.val_mask].argmax(dim=1) == data.y[data.val_mask]).float().mean()
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}')

def run_attack(attack_type, dataset_name, victim_model, data, dataset, device, params):
    generator = GraphGenerator(params['noise_dim'], params['num_nodes'], data.num_features, 
                               generator_type=params['generator_type']).to(device)
    surrogate_model = SurrogateModel(data.num_features, params['hidden_dim'], dataset.num_classes).to(device)

    if attack_type == 1:
        attack = TypeIAttack(generator, surrogate_model, victim_model, device, 
                             params['noise_dim'], params['num_nodes'], data.num_features,
                             generator_lr=params['generator_lr'], surrogate_lr=params['surrogate_lr'],
                             n_generator_steps=params['n_generator_steps'], n_surrogate_steps=params['n_surrogate_steps'])
    elif attack_type == 2:
        attack = TypeIIAttack(generator, surrogate_model, victim_model, device, 
                              params['noise_dim'], params['num_nodes'], data.num_features,
                              generator_lr=params['generator_lr'], surrogate_lr=params['surrogate_lr'],
                              n_generator_steps=params['n_generator_steps'], n_surrogate_steps=params['n_surrogate_steps'])
    elif attack_type == 3:
        surrogate_model2 = SurrogateModel(data.num_features, params['hidden_dim'], dataset.num_classes).to(device)
        attack = TypeIIIAttack(generator, surrogate_model, surrogate_model2, victim_model, device, 
                               params['noise_dim'], params['num_nodes'], data.num_features,
                               generator_lr=params['generator_lr'], surrogate_lr=params['surrogate_lr'],
                               n_generator_steps=params['n_generator_steps'], n_surrogate_steps=params['n_surrogate_steps'])
    else:
        raise ValueError("Invalid attack type. Choose 1, 2, or 3.")

    trained_surrogate, _, _ = attack.attack(params['num_queries'])
    accuracy, fidelity = evaluate_models(victim_model, trained_surrogate, data)
    return accuracy, fidelity

def main():
    parser = argparse.ArgumentParser(description="STEALGNN Interactive Example")
    parser.add_argument("attack_type", type=int, choices=[1, 2, 3], help="Attack type (1, 2, or 3)")
    parser.add_argument("dataset_name", type=str, choices=['cora', 'computers', 'pubmed', 'ogb-arxiv'], help="Dataset name")
    parser.add_argument("--victim_model_path", type=str, help="Path to custom victim model file (optional)")
    args = parser.parse_args()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    
    # Load dataset and create/load victim model
    dataset, data, victim_model = load_dataset_and_create_victim_model(args.dataset_name, device)
    
    if args.victim_model_path:
        try:
            victim_model.load_state_dict(torch.load(args.victim_model_path, map_location=device))
            print(f"Loaded custom victim model from {args.victim_model_path}")
        except FileNotFoundError:
            print(f"Error: Victim model file not found at {args.victim_model_path}")
            print("Training a new victim model instead...")
            train_victim_model(victim_model, data, args.dataset_name)
    else:
        print("Training victim model...")
        train_victim_model(victim_model, data, args.dataset_name)

    # Set default parameters
    params = {
        'noise_dim': 32,
        'num_nodes': 500,
        'hidden_dim': 16,
        'generator_type': 'cosine',
        'generator_lr': 1e-6,
        'surrogate_lr': 0.001,
        'n_generator_steps': 2,
        'n_surrogate_steps': 5,
        'num_queries': 100
    }

    # Allow user to tweak parameters
    print("\nCurrent parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")
    
    change_params = input("\nDo you want to change any parameters? (y/n): ").lower() == 'y'
    if change_params:
        for key in params:
            new_value = input(f"Enter new value for {key} (press Enter to keep current value): ")
            if new_value:
                params[key] = type(params[key])(new_value)

    print("\nRunning attack with the following parameters:")
    for key, value in params.items():
        print(f"{key}: {value}")

    accuracy, fidelity = run_attack(args.attack_type, args.dataset_name, victim_model, data, dataset, device, params)
    
    print(f"\nResults for Type {args.attack_type} Attack on {args.dataset_name} dataset:")
    print(f"Accuracy: {accuracy:.4f}")
    print(f"Fidelity: {fidelity:.4f}")

if __name__ == "__main__":
    main()
