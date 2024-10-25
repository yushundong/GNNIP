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

def evaluate_model(model_output, data):
    if isinstance(model_output, tuple):
        # Handle case where we have two surrogate models
        model1, model2 = model_output
        model1.eval()
        model2.eval()
        with torch.no_grad():
            # Get predictions from both models
            out1 = model1(data.x, data.edge_index)
            out2 = model2(data.x, data.edge_index)
            # Average the predictions
            out = (out1 + out2) / 2
            pred = out.argmax(dim=1)
    else:
        # Handle single model case
        model_output.eval()
        with torch.no_grad():
            out = model_output(data.x, data.edge_index)
            pred = out.argmax(dim=1)
    
    correct = pred[data.test_mask] == data.y[data.test_mask]
    accuracy = int(correct.sum()) / int(data.test_mask.sum())
    return accuracy

def run_attack(victim_model, data, device, attack_type):
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

    if attack_type == 1:
        attack = TypeIAttack(generator, surrogate_model, victim_model, device,
                           noise_dim, num_nodes, feature_dim, generator_lr, surrogate_lr,
                           n_generator_steps, n_surrogate_steps)
    elif attack_type == 2:
        attack = TypeIIAttack(generator, surrogate_model, victim_model, device,
                            noise_dim, num_nodes, feature_dim, generator_lr, surrogate_lr,
                            n_generator_steps, n_surrogate_steps)
    elif attack_type == 3:
        surrogate_model2 = create_victim_model_cora().to(device)
        attack = TypeIIIAttack(generator, surrogate_model, surrogate_model2, victim_model, device,
                            noise_dim, num_nodes, feature_dim, generator_lr, surrogate_lr,
                            n_generator_steps, n_surrogate_steps)
    else:
        raise ValueError("Invalid attack type. Please choose 1, 2, or 3.")

    print(f"\nRunning Type {attack_type} Attack...")
    trained_surrogate, _, _ = attack.attack(num_queries)
    surrogate_accuracy = evaluate_model(trained_surrogate, data)
    print(f"Type {attack_type} Attack - Surrogate Model Accuracy: {surrogate_accuracy:.4f}")

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

    # Get attack type from user
    while True:
        try:
            attack_type = int(input("\nChoose attack type (1, 2, or 3): "))
            if attack_type in [1, 2, 3]:
                break
            else:
                print("Please enter 1, 2, or 3.")
        except ValueError:
            print("Please enter a valid number (1, 2, or 3).")

    # Run selected attack
    run_attack(victim_model, data, device, attack_type)

if __name__ == "__main__":
    main()
