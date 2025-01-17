import networkx as nx
import dgl


def compute_jaccard_coefficient(nx_g, pairs):
    """
    Computes the Jaccard coefficient for each pair of nodes in the graph.

    Args:
        nx_g (networkx.Graph): The graph object.
        pairs (list of tuple): List of node pairs to compute Jaccard coefficient.

    Returns:
        dict: A dictionary where the keys are node pairs and the values are their Jaccard coefficients.
    """
    jaccard_dict = {}
    for u, v, p in nx.jaccard_coefficient(nx_g, pairs):
        jaccard_dict[(u, v)] = round(p, 3)
    return jaccard_dict


def compute_preferential_attachment(nx_g, pairs):
    """
    Computes the preferential attachment for each pair of nodes in the graph.

    Args:
        nx_g (networkx.Graph): The graph object.
        pairs (list of tuple): List of node pairs to compute preferential attachment.

    Returns:
        dict: A dictionary where the keys are node pairs and the values are their preferential attachment scores.
    """
    attach_dict = {}
    for u, v, p in nx.preferential_attachment(nx_g, pairs):
        attach_dict[(u, v)] = round(p, 3)
    return attach_dict


def compute_common_neighbors(nx_g, pairs):
    """
    Computes the number of common neighbors for each pair of nodes in the graph.

    Args:
        nx_g (networkx.Graph): The graph object.
        pairs (list of tuple): List of node pairs to compute common neighbors.

    Returns:
        dict: A dictionary where the keys are node pairs and the values are the number of common neighbors.
    """
    neighbors_dict = {}
    for start_id, end_id in pairs:
        neighbors_dict[(start_id, end_id)] = len(list(nx.common_neighbors(nx_g, start_id, end_id)))
    return neighbors_dict


def generate_graph_features(g, pairs, k=1, label=1):
    """
    Generate graph-based features for a given set of node pairs.

    Args:
        g (DGLGraph): The input DGL graph.
        pairs (list of tuple): List of node pairs for which the features are to be generated.
        k (int): The radius for the ego graph.
        label (int): The label of the edge to be removed for the subgraph when label = 1.

    Returns:
        tuple: A tuple containing three dictionaries for Jaccard, Preferential Attachment, and Common Neighbors.
    """
    nx_g = nx.Graph(dgl.to_networkx(g, node_attrs=["features"]))
    
    jaccard_dict = {}
    attach_dict = {}
    neighbors_dict = {}

    for pair in pairs:
        start_subgraph_nodes = list(nx.ego.ego_graph(nx_g, n=pair[0], radius=k).nodes())
        end_subgraph_nodes = list(nx.ego.ego_graph(nx_g, n=pair[1], radius=k).nodes())
        subgraph_nodes = start_subgraph_nodes + end_subgraph_nodes
        subgraph = nx_g.subgraph(subgraph_nodes).copy()

        if label == 1:
            subgraph.remove_edge(pair[0], pair[1])

        # Compute features for the subgraph
        jaccard_dict[pair] = round(next(nx.jaccard_coefficient(subgraph, [(pair[0], pair[1])]))[2], 3)
        attach_dict[pair] = round(next(nx.preferential_attachment(subgraph, [(pair[0], pair[1])]))[2], 3)
        neighbors_dict[pair] = len(list(nx.common_neighbors(subgraph, pair[0], pair[1])))

    print("Finished generating graph features.")
    return jaccard_dict, attach_dict, neighbors_dict
