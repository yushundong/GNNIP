import dgl
import networkx as nx
from utils.model_query import query_trained_model
from load_data.generate_xy import generate_features


def get_batch(args, batch_pairs, g, k, mode):
    """
    Generates a batch of subgraphs and queries posteriors for node pairs.
    
    Args:
        args: Arguments for the model and batch generation.
        batch_pairs: List of node pairs for which the batch is generated.
        g: The graph object (DGL graph).
        k: The number of hops to consider for subgraphs.
        mode: Mode for querying the trained model.
        
    Returns:
        posteriors_dict_batch: A dictionary of posteriors.
        index_mapping_dict_batch: A dictionary of index mappings for nodes.
    """
    query_graph_batch, index_mapping_dict_batch = get_khop_query_graph_batch(g, batch_pairs, k)
    index_update_batch = [node for _, nodes in index_mapping_dict_batch.items() for node in nodes]
    posteriors_dict_batch = query_trained_model(args, index_update_batch, query_graph_batch, mode)

    print('Finish generating posteriors and mapping dict...')
    return posteriors_dict_batch, index_mapping_dict_batch


def generate_batch_features(args, batch_pairs, g, k, mode, feature_type):
    """
    A generic function to generate features for different types of graph-based batches.
    
    Args:
        args: Arguments for the model and batch generation.
        batch_pairs: List of node pairs for which the batch is generated.
        g: The graph object (DGL graph).
        k: The number of hops to consider for subgraphs.
        mode: Mode for querying the trained model.
        feature_type: The type of features to generate ('node', 'graph', or 'all').
        
    Returns:
        Features, labels, and statistical data for the batch.
    """
    posteriors_dict_batch, index_mapping_dict_batch = get_batch(args, batch_pairs, g, k, mode)

    if feature_type == 'node':
        return generate_features(args, g, batch_pairs, posteriors_dict_batch, index_mapping_dict_batch)
    elif feature_type == 'graph':
        return generate_features(args, g, batch_pairs, posteriors_dict_batch, mode, index_mapping_dict_batch)
    elif feature_type == 'all':
        return generate_features(args, g, batch_pairs, posteriors_dict_batch, mode, index_mapping_dict_batch)
    else:
        return generate_features(args, batch_pairs, posteriors_dict_batch, index_mapping_dict_batch)


def get_khop_query_graph_batch(g, pairs, k=2):
    """
    Generates a k-hop subgraph for each node pair and returns a batched graph.
    
    Args:
        g: The graph object (DGL graph).
        pairs: List of node pairs for which k-hop neighborhoods are generated.
        k: The number of hops to consider for subgraphs.
        
    Returns:
        A batched graph containing the k-hop subgraphs and a mapping of node indices.
    """
    nx_g = dgl.to_networkx(g, node_attrs=["features"])
    subgraph_list = []
    index_mapping_dict = {}
    bias = 0

    for pair in pairs:
        start_node, end_node = pair
        nx_g.remove_edges_from([(start_node, end_node), (end_node, start_node)])
        
        node_index = []
        for node in (start_node, end_node):
            node_neighbors = list(nx.ego.ego_graph(nx_g, n=node, radius=k).nodes())
            node_new_index = node_neighbors.index(node)
            subgraph_k_hop = g.subgraph(node_neighbors)
            subgraph_list.append(subgraph_k_hop)
            node_index.append(node_new_index + bias)
            bias += len(node_neighbors)
        
        nx_g.add_edges_from([(start_node, end_node), (end_node, start_node)])
        index_mapping_dict[(start_node, end_node)] = (node_index[0], node_index[1])

    update_query_graph = dgl.batch(subgraph_list)
    print("Get k-hop query graph")
    return update_query_graph, index_mapping_dict


# Wrapper functions for specific batch feature types
def get_batch_posteriors(args, batch_pairs, g, k, mode):
    batch_features, batch_labels, batch_stat_dict = generate_batch_features(args, batch_pairs, g, k, mode, 'default')
    return batch_features, batch_labels, batch_stat_dict


def get_batch_posteriors_node(args, batch_pairs, g, k, mode):
    batch_node_features, batch_posteriors_features, batch_labels, batch_stat_dict = generate_batch_features(args, batch_pairs, g, k, mode, 'node')
    return batch_node_features, batch_posteriors_features, batch_labels, batch_stat_dict


def get_batch_posteriors_graph(args, batch_pairs, g, k, mode):
    batch_posteriors_features, batch_graph_features, batch_labels, batch_stat_dict = generate_batch_features(args, batch_pairs, g, k, mode, 'graph')
    return batch_posteriors_features, batch_graph_features, batch_labels, batch_stat_dict


def get_batch_posteriors_node_graph(args, batch_pairs, g, k, mode):
    batch_node_features, batch_posteriors_features, batch_graph_features, batch_labels, batch_stat_dict = generate_batch_features(args, batch_pairs, g, k, mode, 'all')
    return batch_node_features, batch_posteriors_features, batch_graph_features, batch_labels, batch_stat_dict
