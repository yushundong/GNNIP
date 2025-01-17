from tqdm import tqdm
import numpy as np
import torch as th
import torch.nn.functional as F
from scipy.spatial import distance
from utils.graph_features import generate_graph_features

# Set random seeds for reproducibility
np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False


def entropy(P: np.ndarray) -> np.ndarray:
    """Calculate the entropy of a probability distribution."""
    epsilon = 1e-5  # Avoid division by zero
    P = P + epsilon
    return np.array([-np.sum(P * np.log(P))])


def js_divergence(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the Jensen-Shannon divergence."""
    return distance.jensenshannon(a, b, base=2.0)


def cosine_sim(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the cosine similarity."""
    return 1 - distance.cosine(a, b)


def correlation_dist(a: np.ndarray, b: np.ndarray) -> float:
    """Calculate the correlation distance."""
    return distance.correlation(a, b)


def pair_wise(a: np.ndarray, b: np.ndarray, edge_feature: str) -> np.ndarray:
    """Generate pair-wise features based on the specified edge feature type."""
    if edge_feature == 'simple':
        return np.concatenate([a, b])
    elif edge_feature == 'add':
        return a + b
    elif edge_feature == 'hadamard':
        return a * b
    elif edge_feature == 'average':
        return (a + b) / 2
    elif edge_feature == 'l1':
        return np.abs(a - b)
    elif edge_feature == 'l2':
        return (a - b) ** 2
    elif edge_feature == 'all':
        return np.concatenate([
            a * b,
            (a + b) / 2,
            np.abs(a - b),
            (a - b) ** 2
        ])
    else:
        raise ValueError(f"Unknown edge feature type: {edge_feature}")


def sim_metrics(a: np.ndarray, b: np.ndarray) -> np.ndarray:
    """Calculate similarity metrics and entropy-based features."""
    a_entropy = entropy(a)
    b_entropy = entropy(b)
    entr_feature = pair_wise(a_entropy, b_entropy, 'all')
    sim_feature = np.array([js_divergence(a, b), cosine_sim(a, b), correlation_dist(a, b)])
    return np.concatenate([entr_feature, sim_feature])


def process_pair(
    start_id: int,
    end_id: int,
    posteriors_dict: dict,
    features: np.ndarray,
    args,
    index_mapping_dict=None
) -> dict:
    """Process a single pair of nodes to generate features."""
    if index_mapping_dict:
        start_id, end_id = index_mapping_dict[(start_id, end_id)]

    start_posterior = F.softmax(posteriors_dict[start_id], dim=0).numpy()
    end_posterior = F.softmax(posteriors_dict[end_id], dim=0).numpy()

    if args.label_only:
        label_dim = len(start_posterior)
        start_label = np.eye(label_dim)[np.argmax(start_posterior)]
        end_label = np.eye(label_dim)[np.argmax(end_posterior)]
        posterior_feature = pair_wise(start_label, end_label, 'add')
    elif args.diff:
        posterior_feature = sim_metrics(start_posterior, end_posterior)
        posterior_feature[np.isnan(posterior_feature)] = 0
    else:
        posterior_feature = pair_wise(start_posterior, end_posterior, args.edge_feature)

    node_feature = None
    if features is not None:
        start_feature = features[start_id].cpu().numpy()
        end_feature = features[end_id].cpu().numpy()
        if args.diff:
            node_feature = sim_metrics(start_feature, end_feature)
            node_feature[np.isnan(node_feature)] = 0
        else:
            node_feature = pair_wise(start_feature, end_feature, 'hadamard')

    return {
        "posterior_feature": posterior_feature,
        "node_feature": node_feature,
        "start_posterior": start_posterior,
        "end_posterior": end_posterior,
    }


def generate_features(
    args,
    pairs,
    posteriors_dict,
    label,
    g=None,
    index_mapping_dict=None,
    mode=None
):
    """Generate features and labels for all pairs."""
    features = g.ndata['features'] if g else None
    stat_dict = {}
    node_features, posterior_features, graph_features = [], [], []
    labels = [label] * len(pairs)

    if mode:
        jaccard_dict, attach_dict, neighbor_dict = generate_graph_features(args, g, pairs, label, mode)

    for start_id, end_id in tqdm(pairs):
        pair_result = process_pair(
            start_id, end_id, posteriors_dict, features, args, index_mapping_dict
        )
        posterior_features.append(pair_result["posterior_feature"])
        if pair_result["node_feature"] is not None:
            node_features.append(pair_result["node_feature"])
        if mode:
            graph_feature = [
                jaccard_dict[(start_id, end_id)],
                attach_dict[(start_id, end_id)],
                neighbor_dict[(start_id, end_id)],
            ]
            graph_features.append(graph_feature)
        stat_dict[(start_id, end_id)] = {
            'node_ids': (start_id, end_id),
            f'{args.node_topology}_start_posterior': pair_result["start_posterior"],
            f'{args.node_topology}_end_posterior': pair_result["end_posterior"],
            f'{args.node_topology}_posterior_feature': pair_result["posterior_feature"],
            'label': label,
        }

    print(f"Features and labels of {len(labels)} pairs have been generated")
    return {
        "node_features": node_features,
        "posterior_features": posterior_features,
        "graph_features": graph_features if mode else None,
        "labels": labels,
        "stat_dict": stat_dict,
    }
