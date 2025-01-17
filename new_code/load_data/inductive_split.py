import numpy as np
import torch as th
import dgl
import os
from tqdm import tqdm
from multiprocessing import Pool
import sys
from utils.model_query import query_trained_model


from load_data.batch_data import get_batch_posteriors
from load_data.generate_xy import generate_features



# Helper functions to avoid redundancy
def make_dirs():
    os.makedirs('./data/pairs/', exist_ok=True)
    os.makedirs('./data/posteriors/', exist_ok=True)
    os.makedirs('./data/mapping/', exist_ok=True)

def generate_pairs(g, train_index):
    start_ids, end_ids = g.edges()
    postive_pairs = []
    negative_pairs = []
    
    for i in tqdm(range(len(start_ids))):
        if start_ids[i] < end_ids[i]:
            postive_pairs.append((start_ids[i].item(), end_ids[i].item()))
    
    num_pos_pairs = len(postive_pairs)
    print(f"There are {num_pos_pairs} edges in the training graph!")
    
    while len(negative_pairs) < num_pos_pairs:
        a, b = np.random.choice(list(train_index), 2, replace=False)
        random_pair = (a, b) if a < b else (b, a)
        if random_pair not in postive_pairs:
            negative_pairs.append(random_pair)
    
    print("Finish Generating Pairs!")
    return postive_pairs, negative_pairs

def remove_neighbor_edge(g):
    """
    Remove all edges from a graph, only save self connection 
    """
    start_ids, end_ids = g.edges()
    delete_eid = [i for i in tqdm(range(len(start_ids))) if start_ids[i] != end_ids[i]]
    g = dgl.remove_edges(g, th.tensor(delete_eid))
    return g

def process_graph_pairs(args, g, mode, positive_pairs, negative_pairs, stat_dict):
    if args.node_topology == '0-hop':
        zero_hop_g = remove_neighbor_edge(g)
        posteriors_dict = query_trained_model(args, np.arange(len(g.nodes())), zero_hop_g, mode)
        
        positive_features, positive_labels, positive_stat_dict = generate_features(args, positive_pairs, posteriors_dict, 1)
        negative_features, negative_labels, negative_stat_dict = generate_features(args, negative_pairs, posteriors_dict, 0)
        
        stat_dict.update({**positive_stat_dict, **negative_stat_dict})
        features = positive_features + negative_features
        labels = positive_labels + negative_labels
        
    elif args.node_topology in ['1-hop', '2-hop']:
        k = 1 if args.node_topology == '1-hop' else 2
        features, labels = [], []
        flag = 1
        for pairs in (positive_pairs, negative_pairs):
            label = flag
            flag -= 1
            batch_size = 4096
            num_batch = len(pairs) // batch_size
            pool = Pool(12)
            results = []
            
            for i in tqdm(range(num_batch+1)):
                batch_pairs = pairs[i*batch_size:(i+1)*batch_size] if i < num_batch else pairs[i*batch_size:]
                batch_result = pool.apply_async(get_batch_posteriors, args=(args, batch_pairs, g, k, i, label, mode))
                results.append(batch_result)
            
            pool.close()
            pool.join()
            for batch_result in results:
                batch_result = batch_result.get()
                features.extend(batch_result[0])
                labels.extend(batch_result[1])
                stat_dict.update(batch_result[2])
    
    return np.array(features).astype(np.float32), th.from_numpy(np.array(features)), labels

def inductive_split(args, train_g, test_g, mode_func):
    make_dirs()
    dataloaders = []
    stat_dicts = []
    for count, g in enumerate([train_g, test_g]):
        mode = f"shadow{str(args.prop)}" if args.prop else "shadow"
        mode = mode if count == 0 else "target"
        if args.diff:
            args.dataset = args.target_dataset if mode == 'target' else args.shadow_dataset
        index = np.arange(len(g.nodes()))
        stat_dict = {}
        
        positive_pairs, negative_pairs = generate_pairs(g, index)
        print(f"Finish Generating Pairs...")
        
        features, features_tensor, labels = mode_func(args, g, mode, positive_pairs, negative_pairs, stat_dict)
        
        indices = th.from_numpy(np.array(positive_pairs + negative_pairs))
        labels = th.tensor(labels)
        
        dataset = th.utils.data.TensorDataset(indices, features_tensor, labels)
        dataloader = th.utils.data.DataLoader(dataset, batch_size=args.batch_size, shuffle=True)
        
        dataloaders.append(dataloader)
        stat_dicts.append(stat_dict)
    
    feature_dim = features[0].shape[0]
    return dataloaders[0], dataloaders[1], feature_dim, stat_dicts[1]


def inductive_split_posteriors(args, train_g, test_g):
    return inductive_split(args, train_g, test_g, process_graph_pairs)

def inductive_split_plus(args, train_g, test_g):
    return inductive_split(args, train_g, test_g, process_graph_pairs)

def inductive_split_plus2(args, train_g, test_g):
    return inductive_split(args, train_g, test_g, process_graph_pairs)

def inductive_split_all(args, train_g, test_g):
    return inductive_split(args, train_g, test_g, process_graph_pairs)
