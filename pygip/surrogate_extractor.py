import torch
import torch.nn.functional as F
from torch_geometric.data import Data
from torch_geometric.nn import GCNConv
import numpy as np


class SurrogateExtractor:
    def __init__(self, input_dim, hidden_dim, output_dim, alpha=0.5):
        """
        Initializes the SurrogateExtractor.

        Parameters:
        input_dim (int): Number of input features per node.
        hidden_dim (int): Number of hidden units.
        output_dim (int): Number of output classes.
        alpha (float): Weight for 2-hop neighbor features in surrogate graph generation.
        """
        self.alpha = alpha
        self.model = GCN(input_dim, hidden_dim, output_dim)
        self.optimizer = torch.optim.Adam(self.model.parameters(), lr=0.01, weight_decay=5e-4)

    def construct_surrogate_graph(self, data, attack_nodes):
        """
        Constructs a surrogate graph by synthesizing features for attack nodes.

        Parameters:
        data (Data): Original graph data.
        attack_nodes (Tensor): Indices of attack nodes.

        Returns:
        Data: Surrogate graph data.
        """
        x = data.x.clone()
        edge_index = data.edge_index.clone()

        for node in attack_nodes:
            neighbors_1hop = edge_index[1][edge_index[0] == node]
            neighbors_2hop = set(
                [edge_index[1][edge_index[0] == neighbor].tolist() for neighbor in neighbors_1hop]
            ) - set(neighbors_1hop.tolist())

            feature_1hop = data.x[neighbors_1hop].mean(dim=0) if len(neighbors_1hop) > 0 else torch.zeros_like(data.x[0])
            feature_2hop = data.x[list(neighbors_2hop)].mean(dim=0) if len(neighbors_2hop) > 0 else torch.zeros_like(data.x[0])

            synthetic_feature = feature_1hop + self.alpha * feature_2hop
            x[node] = synthetic_feature

        return Data(x=x, edge_index=edge_index, y=data.y, train_mask=data.train_mask)

    def train_surrogate_model(self, data):
        """
        Trains the surrogate model using the given graph data.

        Parameters:
        data (Data): Graph data for training.

        Returns:
        GCN: Trained model.
        """
        self.model.train()
        for epoch in range(200):
            self.optimizer.zero_grad()
            out = self.model(data)
            loss = F.nll_loss(out[data.train_mask], data.y[data.train_mask])
            loss.backward()
            self.optimizer.step()
        return self.model

    def evaluate_fidelity(self, target_labels, surrogate_predictions):
        """
        Evaluates the fidelity of the surrogate model.

        Parameters:
        target_labels (Tensor): Ground truth labels from the target model.
        surrogate_predictions (Tensor): Predictions from the surrogate model.

        Returns:
        float: Fidelity score.
        """
        return (target_labels == surrogate_predictions).sum().item() / len(target_labels)

    def fidelity_by_attack_nodes(self, data, attack_node_counts):
        """
        Evaluates fidelity for varying numbers of attack nodes.

        Parameters:
        data (Data): Original graph data.
        attack_node_counts (list): List of attack node counts to evaluate.

        Returns:
        dict: Fidelity scores for each attack node count.
        """
        fidelities = {}

        for count in attack_node_counts:
            attack_nodes = torch.randperm(data.num_nodes)[:count]
            surrogate_data = self.construct_surrogate_graph(data, attack_nodes)
            self.train_surrogate_model(surrogate_data)

            self.model.eval()
            with torch.no_grad():
                surrogate_predictions = self.model(surrogate_data).argmax(dim=1)
                fidelity = self.evaluate_fidelity(data.y, surrogate_predictions)
                fidelities[count] = fidelity
                print(f"Fidelity with {count} attack nodes: {fidelity * 100:.2f}%")

        return fidelities

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)