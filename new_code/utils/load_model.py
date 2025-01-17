import torch as th
import torch.nn.functional as F
from model.gnn import SAGE, GAT, GIN, GCN

def get_gnn_model(config):
    """
    Returns a GNN model based on the given configuration.

    Args:
        config: The configuration object containing model parameters.

    Returns:
        model: The GNN model (SAGE, GAT, GIN, or GCN).
    """
    model_map = {
        'graphsage': SAGE,
        'gat': GAT,
        'gin': GIN,
        'gcn': GCN
    }

    model_class = model_map.get(config.model.lower())

    if model_class is None:
        raise ValueError(f"Unsupported model type: {config.model}")

    return model_class(
        config.in_feats, 
        config.n_hidden, 
        config.n_classes, 
        config.gnn_layers, 
        F.relu, 
        config.batch_size, 
        config.num_workers, 
        config.dropout
    )


def load_model(model, model_path, device):
    """
    Loads a trained model from the given file path.

    Args:
        model: The model to be loaded.
        model_path (str): The path to the saved model file.
        device: The device to load the model onto (CPU or GPU).

    Returns:
        model: The model with loaded weights.
    """
    print(f"Loading model from: {model_path}")
    state_dict = th.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model
