# STEALGNN: Graph Neural Network Model Extraction

This repository contains an implementation of STEALGNN, a framework for data-free model extraction attacks on Graph Neural Networks (GNNs).

## Files

1. `stealgnn.py`: Core implementation of the STEALGNN framework
2. `example.py`: Interactive script to run STEALGNN attacks

## stealgnn.py

This file contains the main implementation of the STEALGNN framework, including:

- `GraphGenerator`: Generates synthetic graphs for the attack
- `SurrogateModel`: The model used to imitate the victim GNN
- `STEALGNN`: Base class for all attack types
- `TypeIAttack`, `TypeIIAttack`, `TypeIIIAttack`: Specific implementations of each attack type
- `evaluate_models`: Function to evaluate the performance of the attacks

## example.py

This interactive script demonstrates how to use the STEALGNN framework. It allows users to:

1. Choose a dataset (Cora, Computers, PubMed, or OGB-Arxiv)
2. Select an attack type (1, 2, or 3)
3. Optionally load a pre-trained victim model
4. Customize attack parameters
5. Run the attack and view results

### Usage

Run the script using the following command:

```
python example.py <attack_type> <dataset_name> [--victim_model_path <path_to_model>]
```

For example:

```
python example.py 1 cora
python example.py 2 pubmed --victim_model_path /path/to/custom_model.pth
```

## Running Experiments from the Original STEALGNN Paper

To exactly replicate the experiments from the original STEALGNN paper, use the following commands and parameters:

### Ensuring Correct Victim Model Architecture

Before running the experiments, modify the `load_dataset_and_create_victim_model` function in `example.py` to use the correct architecture for each dataset:

```python
def load_dataset_and_create_victim_model(dataset_name, device):
    # ... (existing code for loading datasets)

    if dataset_name in ['cora', 'pubmed', 'computers']:
        victim_model = GCN(input_dim, hidden_dim=64, output_dim=dataset.num_classes, num_layers=2).to(device)
    elif dataset_name == 'ogb-arxiv':
        victim_model = GCN(input_dim, hidden_dim=256, output_dim=dataset.num_classes, num_layers=3).to(device)
    else:
        raise ValueError("Invalid dataset name")

    return dataset, data, victim_model

class GCN(torch.nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim, num_layers):
        super(GCN, self).__init__()
        self.convs = torch.nn.ModuleList()
        self.convs.append(GCNConv(input_dim, hidden_dim))
        for _ in range(num_layers - 2):
            self.convs.append(GCNConv(hidden_dim, hidden_dim))
        self.convs.append(GCNConv(hidden_dim, output_dim))

    def forward(self, x, edge_index):
        for conv in self.convs[:-1]:
            x = conv(x, edge_index)
            x = F.relu(x)
            x = F.dropout(x, p=0.5, training=self.training)
        return self.convs[-1](x, edge_index)
```

### Running Experiments

1. For Cora dataset:
   ```
   python example.py 1 cora
   python example.py 2 cora
   python example.py 3 cora
   ```
   Parameters to change:
   - `noise_dim`: 32
   - `num_nodes`: 2485 (Cora's original node count)
   - `hidden_dim`: 64
   - `generator_type`: 'cosine'
   - `generator_lr`: 1e-6
   - `surrogate_lr`: 0.001
   - `n_generator_steps`: 2
   - `n_surrogate_steps`: 5
   - `num_queries`: 700

2. For PubMed dataset:
   ```
   python example.py 1 pubmed
   python example.py 2 pubmed
   python example.py 3 pubmed
   ```
   Parameters to change:
   - `noise_dim`: 32
   - `num_nodes`: 19717 (PubMed's original node count)
   - `hidden_dim`: 64
   - `generator_type`: 'cosine'
   - `generator_lr`: 1e-6
   - `surrogate_lr`: 0.001
   - `n_generator_steps`: 2
   - `n_surrogate_steps`: 5
   - `num_queries`: 700

3. For Amazon Computers dataset:
   ```
   python example.py 1 computers
   python example.py 2 computers
   python example.py 3 computers
   ```
   Parameters to change:
   - `noise_dim`: 32
   - `num_nodes`: 13381 (Amazon Computers' original node count)
   - `hidden_dim`: 64
   - `generator_type`: 'cosine'
   - `generator_lr`: 1e-6
   - `surrogate_lr`: 0.001
   - `n_generator_steps`: 2
   - `n_surrogate_steps`: 5
   - `num_queries`: 700

4. For OGB-Arxiv dataset:
   ```
   python example.py 1 ogb-arxiv
   python example.py 2 ogb-arxiv
   python example.py 3 ogb-arxiv
   ```
   Parameters to change:
   - `noise_dim`: 32
   - `num_nodes`: 169343 (OGB-Arxiv's original node count)
   - `hidden_dim`: 128
   - `generator_type`: 'cosine'
   - `generator_lr`: 1e-6
   - `surrogate_lr`: 0.001
   - `n_generator_steps`: 2
   - `n_surrogate_steps`: 5
   - `num_queries`: 700

For each dataset, run all three attack types (1, 2, and 3) to compare their performance.

When prompted to change parameters in the interactive script, make sure to input the values listed above for each dataset to exactly replicate the paper's experiments.

## Additional Notes

- The experiments use PyTorch Geometric for graph operations and model implementations.
- Adam optimizer is used for both the generator and surrogate model training.
- Run each experiment multiple times (5 times in the original paper) to account for randomness, and average the results for a more robust comparison.
- Ensure that your Python environment has all necessary dependencies installed, including PyTorch, PyTorch Geometric, and OGB (for the OGB-Arxiv dataset).

## Customizing Experiments

When running `example.py`, you'll be prompted to modify the default parameters. You can experiment with different values for:

- `noise_dim`: Dimension of the noise vector for graph generation
- `num_nodes`: Number of nodes in the generated graphs
- `hidden_dim`: Hidden dimension size for the surrogate model
- `generator_type`: Type of graph generator ('cosine' or 'full_param')
- `generator_lr`: Learning rate for the generator
- `surrogate_lr`: Learning rate for the surrogate model
- `n_generator_steps`: Number of generator training steps per iteration
- `n_surrogate_steps`: Number of surrogate model training steps per iteration
- `num_queries`: Total number of queries to the victim model

Adjust these parameters to explore their impact on the attack performance.

By following this README, you should be able to replicate the experiments from the original STEALGNN paper and conduct your own experiments using this implementation.
