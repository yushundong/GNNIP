import torch
import torch.nn as nn
import torch.optim as optim
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures

from models.generator import GraphGenerator
from models.victim import create_victim_model_cora
from attacks.attack1 import TypeIAttack
from attacks.attack2 import TypeIIAttack
from attacks.attack3 import TypeIIIAttack

def train_victim_model(model, data, epochs=200, lr=0.01, weight_decay=5e-4):
    optimizer = optim.Adam(model.parameters(), lr=lr, weight_decay=weight_decay)
    model.train()
    for epoch in range(epochs):
        optimizer.zero_grad()
        out = model(data.x, data.edge_index)
        loss = nn.functional.nll_loss(out[data.train_mask], data.y[data.train_mask])
        loss.backward()
        optimizer.step()
        
        if (epoch + 1) % 10 == 0:
            model.eval()
            with torch.no_grad():
                val_out = model(data.x, data.edge_index)
                val_loss = nn.functional.nll_loss(val_out[data.val_mask], data.y[data.val_mask])
                val_acc = (val_out[data.val_mask].argmax(dim=1) == data.y[data.val_mask]).float().mean()
            model.train()
            print(f'Epoch {epoch+1}/{epochs}, Train Loss: {loss.item():.4f}, Val Loss: {val_loss.item():.4f}, Val Acc: {val_acc.item():.4f}')

def evaluate_model(model, data):
    model.eval()
    with torch.no_grad():
        out = model(data.x, data.edge_index)
        pred = out.argmax(dim=1)
        correct = pred[data.test_mask] == data.y[data.test_mask]
        accuracy = int(correct.sum()) / int(data.test_mask.sum())
    return accuracy

def run_attacks(victim_model, data, device):
    # Initialize generator and surrogate model
    noise_dim = 32
    num_nodes = 500
    feature_dim = data.num_features
    output_dim = data.y.max().item() + 1  # Calculate number of classes

    generator = GraphGenerator(noise_dim, num_nodes, feature_dim, generator_type='cosine').to(device)
    surrogate_model = create_victim_model_cora().to(device)

    # Attack parameters
    num_queries = 300
    generator_lr = 1e-6
    surrogate_lr = 0.001
    n_generator_steps = 2
    n_surrogate_steps = 5

    # Run attacks
    attacks = [
        ("Type I", TypeIAttack(generator, surrogate_model, victim_model, device, 
                               noise_dim, num_nodes, feature_dim, generator_lr, surrogate_lr,
                               n_generator_steps, n_surrogate_steps))]
    '''
        ("Type II", TypeIIAttack(generator, surrogate_model, victim_model, device, 
                                 noise_dim, num_nodes, feature_dim, generator_lr, surrogate_lr,
                                 n_generator_steps, n_surrogate_steps)),
        ("Type III", TypeIIIAttack(generator, surrogate_model, create_victim_model_cora().to(device), 
                                   victim_model, device, noise_dim, num_nodes, feature_dim, generator_lr, surrogate_lr,
                                   n_generator_steps, n_surrogate_steps))
    '''

    for attack_name, attack in attacks:
        print(f"\nRunning {attack_name} Attack...")
        trained_surrogate, _, _ = attack.attack(num_queries)
        surrogate_accuracy = evaluate_model(trained_surrogate, data)
        print(f"{attack_name} Attack - Surrogate Model Accuracy: {surrogate_accuracy:.4f}")

def main():
    # Set up device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load Cora dataset
    dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
    data = dataset[0].to(device)

    # Create and train victim model
    victim_model = create_victim_model_cora().to(device)
    train_victim_model(victim_model, data)

    # Evaluate victim model
    victim_accuracy = evaluate_model(victim_model, data)
    print(f"Victim Model Accuracy: {victim_accuracy:.4f}")

    # Run attacks
    run_attacks(victim_model, data, device)

if __name__ == "__main__":
    main()
