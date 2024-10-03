import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.transforms import NormalizeFeatures
from torch_geometric.nn import GCNConv

# Import the STEALGNN components
from attacks import GraphGenerator, run_attack

class GCN(nn.Module):
    def __init__(self, in_channels, hidden_channels, out_channels):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(in_channels, hidden_channels)
        self.conv2 = GCNConv(hidden_channels, out_channels)
        self.in_channels = in_channels
        self.hidden_channels = hidden_channels
        self.out_channels = out_channels

    def forward(self, x, edge_index):
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=0.5, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

# Set up device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

# Load Cora dataset
dataset = Planetoid(root='/tmp/Cora', name='Cora', transform=NormalizeFeatures())
data = dataset[0].to(device)

# Set up victim model (GCN as described in the paper)
victim_model = GCN(dataset.num_features, 128, dataset.num_classes).to(device)

# Train victim model (you would typically do this separately)
optimizer = torch.optim.Adam(victim_model.parameters(), lr=0.01, weight_decay=5e-4)
victim_model.train()
for epoch in range(200):
    optimizer.zero_grad()
    out = victim_model(data.x, data.edge_index)
    loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
    loss.backward()
    optimizer.step()

# Set up generator and surrogate model for the attack
noise_dim = 32
num_nodes = 500  # As per the paper
feature_dim = dataset.num_features
generator = GraphGenerator(noise_dim, num_nodes, feature_dim, generator_type='cosine').to(device)
surrogate_model = GCN(feature_dim, 128, dataset.num_classes).to(device)

# Run Type 3 attack
num_queries = 400  # As per the paper
attack_type = 3

trained_surrogate, generator_losses, surrogate_losses = run_attack(
    attack_type, generator, surrogate_model, victim_model, num_queries, device, 
    noise_dim, num_nodes, feature_dim
)

# Evaluate the attack
victim_model.eval()
if isinstance(trained_surrogate, tuple):
    surrogate_model1, surrogate_model2 = trained_surrogate
    surrogate_model1.eval()
    surrogate_model2.eval()
else:
    surrogate_model = trained_surrogate
    surrogate_model.eval()

with torch.no_grad():
    victim_output = victim_model(data.x, data.edge_index)
    if isinstance(trained_surrogate, tuple):
        surrogate_output1 = surrogate_model1(data.x, data.edge_index)
        surrogate_output2 = surrogate_model2(data.x, data.edge_index)
        surrogate_output = (surrogate_output1 + surrogate_output2) / 2
    else:
        surrogate_output = surrogate_model(data.x, data.edge_index)

    victim_preds = victim_output.argmax(dim=1)
    surrogate_preds = surrogate_output.argmax(dim=1)

accuracy = (surrogate_preds[data.test_mask] == data.y[data.test_mask]).float().mean()
fidelity = (surrogate_preds == victim_preds).float().mean()

print(f"Attack Type: {attack_type}")
print(f"Accuracy: {accuracy.item():.4f}")
print(f"Fidelity: {fidelity.item():.4f}")

# Plot losses
import matplotlib.pyplot as plt

plt.figure(figsize=(10, 5))
plt.plot(generator_losses, label='Generator Loss')
plt.plot(surrogate_losses, label='Surrogate Loss')
plt.title(f'Losses over time - Type {attack_type} Attack')
plt.xlabel('Query')
plt.ylabel('Loss')
plt.legend()
plt.savefig(f'losses_type{attack_type}.png')
plt.close()
