import numpy as np
import torch as th
import dgl
import networkx as nx
from tqdm import tqdm
from dgl.data import CiteseerGraphDataset, CoraGraphDataset, PubmedGraphDataset

# Set random seeds for reproducibility
np.random.seed(0)
th.manual_seed(0)
th.backends.cudnn.deterministic = True
th.backends.cudnn.benchmark = False

def load_dataset(dataset_name):
    if dataset_name == 'citeseer':
        data = CiteseerGraphDataset()
    elif dataset_name == 'cora':
        data = CoraGraphDataset()
    elif dataset_name == 'pubmed':
        data = PubmedGraphDataset()
    else:
        raise ValueError("Unsupported dataset name")

    graph = data[0]
    nx_g = nx.Graph(graph.to_networkx())
    
    for node_id in nx_g.nodes():
        nx_g.nodes[node_id]["features"] = graph.ndata['feat'][node_id].numpy()
        nx_g.nodes[node_id]["labels"] = graph.ndata['label'][node_id].item()

    dgl_graph = dgl.from_networkx(nx_g, node_attrs=['features', 'labels'])
    dgl_graph = dgl.add_self_loop(dgl_graph)
    dgl_graph = dgl.to_simple(dgl_graph, copy_ndata=True)
    dgl_graph = dgl.to_bidirected(dgl_graph, copy_ndata=True)

    print(f"Graph density: {nx.density(nx_g)}")
    print(f"Classes: {len(set(graph.ndata['label'].tolist()))}")
    print(f"Feature dim: {graph.ndata['feat'].shape[1]}")
    print(f"Graph has {dgl_graph.number_of_nodes()} nodes and {dgl_graph.number_of_edges()} edges.")

    return dgl_graph, len(set(graph.ndata['label'].tolist()))

def node_sample(g, prop=0.5):
    node_number = g.number_of_nodes()
    node_indices = np.random.permutation(node_number)
    split_length = int(node_number * prop)
    return np.sort(node_indices[:split_length]), np.sort(node_indices[split_length:])

def remove_neighbor_edge_by_prop(g, prop=0.2):
    real_pairs = [(i, start.item(), end.item()) for i, (start, end) in enumerate(zip(*g.edges())) if start < end]
    delete_edge_num = int(len(real_pairs) * prop)

    print(f"Real Pairs Number (no self-loop & reverse edge): {len(real_pairs)}")
    print(f"Delete real pairs number (no self-loop & reverse edge): {delete_edge_num}")

    delete_ids = np.random.choice(len(real_pairs), delete_edge_num, replace=False)
    delete_eids = [g.edge_ids(start, end) for i in delete_ids for start, end in real_pairs[i][1:]]

    g = dgl.remove_edges(g, th.tensor(delete_eids))

    print(f"Deleted {len(delete_eids)} edges")
    print(f"Remaining edges: {g.number_of_edges()}")
    return g, [(real_pairs[i][1], real_pairs[i][2]) for i in delete_ids]

def split_target_shadow(g):
    target_index, shadow_index = node_sample(g)
    return g.subgraph(target_index), g.subgraph(shadow_index)

def split_target_shadow_by_prop(args, g):
    target_g, shadow_g = split_target_shadow(g)
    shadow_index_prop, _ = node_sample(shadow_g, args.prop * 0.01)
    shadow_g = shadow_g.subgraph(shadow_index_prop)
    return target_g, shadow_g

def split_train_test(g):
    train_index, test_index = node_sample(g, 0.8)
    return g.subgraph(train_index), g.subgraph(test_index)
