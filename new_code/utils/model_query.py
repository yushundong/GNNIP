import os
import torch as th
from utils.load_model import load_model, get_gnn_model

def query_trained_model(config, train_index, graph, mode):
    """
    Query a trained model using the 0-hop training graph nodes.

    Args:
        config: The configuration object containing model and dataset parameters.
        train_index (list): List of training node indices.
        graph (DGLGraph): The graph used for inference.
        mode (str): The mode in which the model is used ('target' or 'shadow').

    Returns:
        dict: A dictionary with node indices as keys and model predictions as values.
    """
    # Set the features and classes according to the mode ('target' or 'shadow')
    if config.diff:
        config.in_feats = config.target_in_feats if mode == 'target' else config.shadow_in_feats
        config.n_classes = config.target_classes if mode == 'target' else config.shadow_classes

    # Set the model according to the mode
    config.model = config.target_model if mode == 'target' else config.shadow_model
    # config.device = th.device('cpu')  # Set device to CPU explicitly
    config.device = th.device('cuda' if th.cuda.is_available() else 'cpu')

    model = get_gnn_model(config).to(config.device)

    model_save_path = os.path.join(config.model_save_path, f'{config.setting}_{config.dataset}_{config.model}_{mode}.pth')
    print(f"Loading {mode} model from: {model_save_path}")
    model = load_model(model, model_save_path, config.device)

    # Query the trained model
    model.eval()
    with th.no_grad():
        predictions = model.inference(graph, graph.ndata['features'], config.device)

    # Store the predictions for the training nodes
    result_dict = {train_index[i]: predictions[train_index[i]] for i in range(len(train_index))}

    print(f"Finished querying {mode} model!")
    return result_dict
