import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch_geometric.nn import GCNConv
from torch_geometric.utils import to_dense_adj
import numpy as np
from tqdm import tqdm

class GraphGenerator(nn.Module):
    def __init__(self, noise_dim, num_nodes, feature_dim, generator_type='cosine', threshold=0.1):
        super(GraphGenerator, self).__init__()
        self.noise_dim = noise_dim
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim
        self.generator_type = generator_type
        self.threshold = threshold
       
        self.feature_gen = nn.Sequential(
            nn.Linear(noise_dim, 128),
            nn.ReLU(),
            nn.Linear(128, 256),
            nn.ReLU(),
            nn.Linear(256, num_nodes * feature_dim),
            nn.Tanh()
        )
       
        if generator_type == 'full_param':
            self.structure_gen = nn.Sequential(
                nn.Linear(noise_dim, 128),
                nn.ReLU(),
                nn.Linear(128, 256),
                nn.ReLU(),
                nn.Linear(256, num_nodes * num_nodes),
                nn.Sigmoid()
            )

    def forward(self, z):
        features = self.feature_gen(z).view(self.num_nodes, self.feature_dim)
       
        if self.generator_type == 'cosine':
            adj = self.cosine_similarity_generator(features)
        elif self.generator_type == 'full_param':
            adj = self.full_param_generator(z)
        else:
            raise ValueError("Invalid generator type. Choose 'cosine' or 'full_param'.")
        
        adj = adj / adj.sum(1, keepdim=True).clamp(min=1)
        
        return features, adj

    def cosine_similarity_generator(self, features):
        norm_features = F.normalize(features, p=2, dim=1)
        adj = torch.mm(norm_features, norm_features.t())
        adj = (adj > self.threshold).float()
        adj = adj * (1 - torch.eye(self.num_nodes, device=adj.device))
        return adj

    def full_param_generator(self, z):
        adj = self.structure_gen(z).view(self.num_nodes, self.num_nodes)
        adj = (adj + adj.t()) / 2
        adj = adj * (1 - torch.eye(self.num_nodes, device=adj.device))
        return adj

    def adj_to_edge_index(self, adj):
        return adj.nonzero().t()

class SurrogateModel(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers=2):
        super(SurrogateModel, self).__init__()
        self.convs = nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for i, conv in enumerate(self.convs[:-1]):
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        
        x = self.convs[-1](x, edge_index)
        return F.log_softmax(x, dim=1)

class STEALGNN:
    def __init__(self, generator, surrogate_model, victim_model, device, 
                 noise_dim, num_nodes, feature_dim,
                 generator_lr=1e-6, surrogate_lr=0.001,
                 n_generator_steps=2, n_surrogate_steps=5):
        self.generator = generator
        self.surrogate_model = surrogate_model
        self.victim_model = victim_model
        self.device = device
        self.noise_dim = noise_dim
        self.num_nodes = num_nodes
        self.feature_dim = feature_dim

        self.generator_optimizer = optim.Adam(self.generator.parameters(), lr=generator_lr)
        self.surrogate_optimizer = optim.Adam(self.surrogate_model.parameters(), lr=surrogate_lr)
        
        self.criterion = nn.CrossEntropyLoss()
        self.n_generator_steps = n_generator_steps
        self.n_surrogate_steps = n_surrogate_steps

    def generate_graph(self):
        z = torch.randn(1, self.noise_dim).to(self.device)
        features, adj = self.generator(z)
        edge_index = self.generator.adj_to_edge_index(adj)
        return features, edge_index

    def train_surrogate(self):
        self.generator.eval()
        self.surrogate_model.train()

        total_loss = 0
        for _ in range(self.n_surrogate_steps):
            self.surrogate_optimizer.zero_grad()
            
            features, edge_index = self.generate_graph()

            with torch.no_grad():
                victim_output = self.victim_model(features, edge_index)
            surrogate_output = self.surrogate_model(features, edge_index)

            loss = self.criterion(surrogate_output, victim_output.argmax(dim=1))
            
            loss.backward()
            self.surrogate_optimizer.step()

            total_loss += loss.item()

        return total_loss / self.n_surrogate_steps

    def attack(self, num_queries):
        generator_losses = []
        surrogate_losses = []

        pbar = tqdm(range(num_queries), desc=f"Running {self.__class__.__name__}")
        for _ in pbar:
            gen_loss = self.train_generator()
            surr_loss = self.train_surrogate()

            generator_losses.append(gen_loss)
            surrogate_losses.append(surr_loss)

            pbar.set_postfix({'Gen Loss': f"{gen_loss:.4f}", 'Surr Loss': f"{surr_loss:.4f}"})

        return self.surrogate_model, generator_losses, surrogate_losses

class TypeIAttack(STEALGNN):
    def train_generator(self):
        self.generator.train()
        self.surrogate_model.eval()

        total_loss = 0
        for _ in range(self.n_generator_steps):
            self.generator_optimizer.zero_grad()
            
            features, edge_index = self.generate_graph()

            with torch.no_grad():
                victim_output = self.victim_model(features, edge_index)
            surrogate_output = self.surrogate_model(features, edge_index)

            loss = -self.criterion(surrogate_output, victim_output.argmax(dim=1))

            epsilon = 1e-6
            num_directions = 2
            estimated_gradient = torch.zeros_like(features)
            
            for _ in range(num_directions):
                u = torch.randn_like(features)
                perturbed_features = features + epsilon * u
                
                with torch.no_grad():
                    perturbed_victim_output = self.victim_model(perturbed_features, edge_index)
                perturbed_surrogate_output = self.surrogate_model(perturbed_features, edge_index)
                perturbed_loss = -self.criterion(perturbed_surrogate_output, perturbed_victim_output.argmax(dim=1))
                
                estimated_gradient += (perturbed_loss - loss) / epsilon * u
            
            estimated_gradient /= num_directions
            features.grad = estimated_gradient

            self.generator_optimizer.step()
            total_loss += loss.item()

        return total_loss / self.n_generator_steps

class TypeIIAttack(STEALGNN):
    def train_generator(self):
        self.generator.train()
        self.surrogate_model.eval()

        total_loss = 0
        for _ in range(self.n_generator_steps):
            self.generator_optimizer.zero_grad()
            
            features, edge_index = self.generate_graph()

            with torch.no_grad():
                victim_output = self.victim_model(features, edge_index)
            surrogate_output = self.surrogate_model(features, edge_index)

            loss = -self.criterion(surrogate_output, victim_output.argmax(dim=1))
            loss.backward()

            self.generator_optimizer.step()
            total_loss += loss.item()

        return total_loss / self.n_generator_steps

class TypeIIIAttack(STEALGNN):
    def __init__(self, generator, surrogate_model1, surrogate_model2, victim_model, device, 
                 noise_dim, num_nodes, feature_dim,
                 generator_lr=1e-6, surrogate_lr=0.001,
                 n_generator_steps=2, n_surrogate_steps=5):
        super().__init__(generator, surrogate_model1, victim_model, device, 
                         noise_dim, num_nodes, feature_dim,
                         generator_lr, surrogate_lr,
                         n_generator_steps, n_surrogate_steps)
        self.surrogate_model2 = surrogate_model2
        self.surrogate_optimizer2 = optim.Adam(self.surrogate_model2.parameters(), lr=surrogate_lr)

    def train_generator(self):
        self.generator.train()
        self.surrogate_model.eval()
        self.surrogate_model2.eval()

        total_loss = 0
        for _ in range(self.n_generator_steps):
            self.generator_optimizer.zero_grad()
            
            features, edge_index = self.generate_graph()

            surrogate_output1 = self.surrogate_model(features, edge_index)
            surrogate_output2 = self.surrogate_model2(features, edge_index)

            loss = -torch.mean(torch.std(torch.stack([surrogate_output1, surrogate_output2]), dim=0))
            loss.backward()

            self.generator_optimizer.step()
            total_loss += loss.item()

        return total_loss / self.n_generator_steps

    def train_surrogate(self):
        self.generator.eval()
        self.surrogate_model.train()
        self.surrogate_model2.train()

        total_loss = 0
        for _ in range(self.n_surrogate_steps):
            self.surrogate_optimizer.zero_grad()
            self.surrogate_optimizer2.zero_grad()
            
            features, edge_index = self.generate_graph()

            with torch.no_grad():
                victim_output = self.victim_model(features, edge_index)
            surrogate_output1 = self.surrogate_model(features, edge_index)
            surrogate_output2 = self.surrogate_model2(features, edge_index)

            loss1 = self.criterion(surrogate_output1, victim_output.argmax(dim=1))
            loss2 = self.criterion(surrogate_output2, victim_output.argmax(dim=1))
            
            combined_loss = loss1 + loss2
            combined_loss.backward()
            
            self.surrogate_optimizer.step()
            self.surrogate_optimizer2.step()

            total_loss += combined_loss.item() / 2

        return total_loss / self.n_surrogate_steps

def evaluate_models(victim_model, surrogate_model, data):
    victim_model.eval()
    surrogate_model.eval()
    
    with torch.no_grad():
        victim_out = victim_model(data.x, data.edge_index)
        surrogate_out = surrogate_model(data.x, data.edge_index)
        
        victim_preds = victim_out.argmax(dim=1)
        surrogate_preds = surrogate_out.argmax(dim=1)

    accuracy = (surrogate_preds[data.test_mask] == data.y[data.test_mask]).float().mean().item()
    fidelity = (surrogate_preds[data.test_mask] == victim_preds[data.test_mask]).float().mean().item()

    return accuracy, fidelity
