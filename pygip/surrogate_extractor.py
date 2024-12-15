import torch
import torch.nn.functional as F
from torch_geometric.datasets import Planetoid
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data
import numpy as np
import matplotlib.pyplot as plt

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, dropout_rate=0.5):
        super(GCN, self).__init__()
        self.conv1 = GCNConv(input_dim, hidden_dim)
        self.conv2 = GCNConv(hidden_dim, output_dim)
        self.dropout_rate = dropout_rate

    def forward(self, data):
        x, edge_index = data.x, data.edge_index
        x = F.relu(self.conv1(x, edge_index))
        x = F.dropout(x, p=self.dropout_rate, training=self.training)
        x = self.conv2(x, edge_index)
        return F.log_softmax(x, dim=1)

class SurrogateAttack:
    def __init__(self, dataset_name, hidden_dim=32, dropout_rate=0.5, alpha=0.5, epochs=300):
        self.dataset_name = dataset_name
        self.hidden_dim = hidden_dim
        self.dropout_rate = dropout_rate
        self.alpha = alpha
        self.epochs = epochs
        self.dataset = Planetoid(root='/tmp/Planetoid', name=dataset_name)
        self.data = self.dataset[0]
        self.input_dim = self.data.x.shape[1]
        self.output_dim = self.dataset.num_classes

    def train_model(self, data):
        model = GCN(self.input_dim, self.hidden_dim, self.output_dim, dropout_rate=self.dropout_rate)
        optimizer = torch.optim.Adam(model.parameters(), lr=0.01, weight_decay=5e-4)
        model.train()
        for epoch in range(self.epochs):
            optimizer.zero_grad()
            out = model(data)
            valid_train_mask = data.train_mask & (data.y != -1)
            loss = F.nll_loss(out[valid_train_mask], data.y[valid_train_mask])
            loss.backward()
            optimizer.step()
        return model

    def construct_surrogate_graph(self, attack_nodes):
        x = self.data.x.clone()
        edge_index = self.data.edge_index.clone()
        train_edges = self.data.train_mask[edge_index[0]] & self.data.train_mask[edge_index[1]]
        edge_index = edge_index[:, train_edges]
        degrees = torch.bincount(edge_index[0], minlength=self.data.num_nodes)

        synthetic_nodes = []
        synthetic_edges = []

        for node in attack_nodes:
            neighbors_1hop = edge_index[1][edge_index[0] == node]
            neighbors_1hop = neighbors_1hop[self.data.train_mask[neighbors_1hop]]

            neighbors_2hop = set()
            for neighbor in neighbors_1hop:
                neighbors_of_neighbor = edge_index[1][edge_index[0] == neighbor]
                neighbors_2hop.update(neighbors_of_neighbor[self.data.train_mask[neighbors_of_neighbor]].tolist())
            neighbors_2hop -= set(neighbors_1hop.tolist())

            synthetic_node_id = len(x)
            synthetic_nodes.append(synthetic_node_id)
            synthetic_edges.extend([[node, synthetic_node_id], [synthetic_node_id, node]])

            feature_1hop = sum(self.data.x[neighbor] / degrees[neighbor] for neighbor in neighbors_1hop) / len(neighbors_1hop) if len(neighbors_1hop) > 0 else torch.zeros_like(self.data.x[0])
            feature_2hop = sum(self.data.x[neighbor] / degrees[neighbor] for neighbor in neighbors_2hop) / len(neighbors_2hop) if len(neighbors_2hop) > 0 else torch.zeros_like(self.data.x[0])
            synthetic_feature = feature_1hop + self.alpha * feature_2hop

            x = torch.cat([x, synthetic_feature.unsqueeze(0)], dim=0)

        if synthetic_edges:
            synthetic_edges = torch.tensor(synthetic_edges, dtype=torch.long).t().contiguous()
            edge_index = torch.cat([edge_index, synthetic_edges], dim=1)

        new_train_mask = torch.cat([self.data.train_mask, torch.zeros(len(synthetic_nodes), dtype=torch.bool)])
        new_y = torch.cat([self.data.y, torch.full((len(synthetic_nodes),), -1)])

        return Data(x=x, edge_index=edge_index, y=new_y, train_mask=new_train_mask)

    def evaluate_fidelity(self, target_model, surrogate_model):
        target_model.eval()
        surrogate_model.eval()
        with torch.no_grad():
            target_preds = target_model(self.data).argmax(dim=1)
            surrogate_preds = surrogate_model(self.data).argmax(dim=1)
            return (target_preds == surrogate_preds).sum().item() / len(target_preds)

    def run_attack(self, attack_node_counts):
        target_model = self.train_model(self.data)
        fidelities = []
        degrees = torch.bincount(self.data.edge_index[0], minlength=self.data.num_nodes)

        for count in attack_node_counts:
            threshold = torch.quantile(degrees.float(), 0.5)
            low_degree_nodes = (degrees <= threshold).nonzero(as_tuple=True)[0]

            if len(low_degree_nodes) >= count:
                attack_nodes = low_degree_nodes[torch.randperm(len(low_degree_nodes))[:count]]
            else:
                print(f"Warning: Not enough low-degree nodes to select {count} attack nodes. Selecting all available.")
                attack_nodes = low_degree_nodes

            surrogate_data = self.construct_surrogate_graph(attack_nodes)
            surrogate_model = self.train_model(surrogate_data)
            fidelity = self.evaluate_fidelity(target_model, surrogate_model)
            fidelities.append(fidelity)
            print(f"Fidelity with {count} attack nodes: {fidelity * 100:.2f}%")

        return fidelities