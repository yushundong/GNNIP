import os
import torch as th
import networkx as nx
import datetime
import argparse
import numpy as np
import random

from load_data.load_graph import load_dataset, split_target_shadow, split_target_shadow_by_prop, split_train_test
from load_data.inductive_split import inductive_split_posteriors, inductive_split_plus, inductive_split_plus2, inductive_split_all
from attack.attack import run_attack, run_attack_three_features, run_attack_two_features


th.set_num_threads(1)

def arg_parse():
    argparser = argparse.ArgumentParser("multi-gpu training")
    argparser.add_argument('--gpu', type=int, default=-1, help="GPU device ID. Use -1 for CPU training")
    argparser.add_argument('--dataset', type=str, default='Cora')
    argparser.add_argument('--node_topology', type=str, default='0-hop', help="node topology used to query the model 0-hop, 2-hop")
    argparser.add_argument('--num_epochs', type=int, default=200)
    argparser.add_argument('--edge_feature', type=str, default='all')
    argparser.add_argument('--n_hidden', type=int, default=128)
    argparser.add_argument('--mlp_layers', type=int, default=3)
    argparser.add_argument('--gnn_layers', type=int, default=2)
    argparser.add_argument('--batch_size', type=int, default=1000)
    argparser.add_argument('--lr', type=float, default=0.001)
    argparser.add_argument('--log-every', type=int, default=20)
    argparser.add_argument('--eval-every', type=int, default=10)   
    argparser.add_argument('--dropout', type=float, default=0.5)
    argparser.add_argument("--seed", type=int, default=0, help="seed")
    argparser.add_argument('--optim', type=str, default='adam')
    argparser.add_argument('--target_model', type=str, default='graphsage')
    argparser.add_argument('--shadow_model', type=str, default='graphsage')
    argparser.add_argument('--num_workers', type=int, default=0, help="Number of sampling processes. Use 0 for no extra process.")
    argparser.add_argument('--model_save_path', type=str, default='../data/save_model/gnn/')
    argparser.add_argument('--attack_model_save_path', type=str, default='../data/save_model/mlp/')
    argparser.add_argument('--load_trained', type=str, default='no')
    argparser.add_argument('--plus', action='store_true')
    argparser.add_argument('--plus2', action='store_true')
    argparser.add_argument('--all', action='store_true')
    argparser.add_argument('--scheduler', action='store_true')
    argparser.add_argument('--perturb_type', type=str, default='discrete')
    argparser.add_argument('--dp', action='store_true')
    argparser.add_argument('--epsilon', type=int, default=8)
    argparser.add_argument('--label_only', action='store_true')
    argparser.add_argument('--soft_prob', action='store_true')
    argparser.add_argument('--T', type=int, default=20)
    argparser.add_argument('--prop', type=int, help="use a specified proportion of the shadow dataset")
    args = argparser.parse_args()

    # args.device = th.device('cuda:%d' % args.gpu if args.gpu >= 0 else 'cpu')
    args.device = th.device('cpu')
    args.trad = False
    return args

def initialize_seeds(args):
    random.seed(args.seed)
    np.random.seed(args.seed)
    th.manual_seed(args.seed)

def prepare_graph_data(args):
    g, n_classes = load_dataset(args.dataset)
    print(nx.density(g.to_networkx()))

    args.diff = False
    args.in_feats = g.ndata['features'].shape[1]
    args.node_feature_dim = args.in_feats
    args.graph_feature_dim = 3
    args.n_classes = n_classes
    args.setting = 'inductive'

    if args.prop:
        target_g, shadow_g = split_target_shadow_by_prop(args, g)
    else:
        target_g, shadow_g = split_target_shadow(g)

    target_train_g, target_test_g = split_train_test(target_g)
    shadow_train_g, shadow_test_g = split_train_test(shadow_g)
    target_train_g.create_formats_()
    shadow_train_g.create_formats_()

    return target_train_g, shadow_train_g, n_classes

def print_graph_statistics(target_train_g, shadow_train_g, n_classes, args):
    print(f"Target Train Graph Num of Edges: {len(target_train_g.edges()[0])}")
    print(f"Target Train Graph Num of Nodes: {len(target_train_g.nodes())}")
    print(f"Target Train Graph Density: {nx.density(target_train_g.to_networkx()):.5f}")

    print(f"Shadow Train Graph Num of Edges: {len(shadow_train_g.edges()[0])}")
    print(f"Shadow Train Graph Num of Nodes: {len(shadow_train_g.nodes())}")
    print(f"Shadow Train Graph Density: {nx.density(shadow_train_g.to_networkx()):.5f}")
    
    print(f"Classes: {n_classes}")
    print(f"Feature dim: {args.in_feats}")

def perform_attack(args, target_train_g, shadow_train_g):
    if args.plus:
        return perform_plus_attack(args, target_train_g, shadow_train_g)
    elif args.plus2:
        return perform_plus2_attack(args, target_train_g, shadow_train_g)
    elif args.all:
        return perform_all_attack(args, target_train_g, shadow_train_g)
    else:
        return perform_default_attack(args, target_train_g, shadow_train_g)

def perform_plus_attack(args, target_train_g, shadow_train_g):
    args.feature = 'label_node' if args.label_only else 'posteriors_node' 
    args.method = f'{args.node_topology}_{args.feature}' 
    train_dataloader, test_dataloader, posterior_feature_dim, stat_dict = inductive_split_plus(args, shadow_train_g, target_train_g)
    return run_attack_two_features(args, posterior_feature_dim, train_dataloader, test_dataloader, stat_dict)

def perform_plus2_attack(args, target_train_g, shadow_train_g):
    args.feature = 'label_graph' if args.label_only else 'posteriors_graph'
    args.method = f'{args.node_topology}_{args.feature}' 
    train_dataloader, test_dataloader, posterior_feature_dim, stat_dict = inductive_split_plus2(args, shadow_train_g, target_train_g)
    return run_attack_two_features(args, posterior_feature_dim, train_dataloader, test_dataloader, stat_dict)

def perform_all_attack(args, target_train_g, shadow_train_g):
    args.feature = 'label_node_graph' if args.label_only else 'posteriors_node_graph' 
    args.method = f'{args.node_topology}_{args.feature}'
    train_dataloader, test_dataloader, posterior_feature_dim, stat_dict = inductive_split_all(args, shadow_train_g, target_train_g)
    return run_attack_three_features(args, posterior_feature_dim, train_dataloader, test_dataloader, stat_dict)

def perform_default_attack(args, target_train_g, shadow_train_g):
    args.feature = 'label' if args.label_only else 'posteriors'
    args.method = f'{args.node_topology}_{args.feature}'
    train_dataloader, test_dataloader, feature_dim, stat_dict = inductive_split_posteriors(args, shadow_train_g, target_train_g)
    return run_attack(args, feature_dim, train_dataloader, test_dataloader, stat_dict)

def save_results(args, stat_dict, log_dir, k, train_acc, train_auc, test_acc, test_auc):
    pickle_path = os.path.join(args.data_save_path, f'{args.setting}_{args.dataset}_{args.target_model}_{args.shadow_model}_{args.method}.pickle')
    th.save(stat_dict, pickle_path)
    
    is_scheduled = 1 if args.scheduler else 0
    with open(os.path.join(log_dir, "attack_performance.txt"), "a") as wf:
        wf.write(f"{args.dataset}, {args.target_model}, {args.shadow_model}, {args.edge_feature}, {is_scheduled}, {args.optim}, {args.lr}, {args.method}, {train_acc:.3f}, {train_auc:.3f}, {test_acc:.3f}, {test_auc:.3f}, {str(datetime.timedelta(seconds=k))}, {args.seed}\n")

def main():
    args = arg_parse()
    args.model_save_path = './data/save_model/gnn/'
    args.data_save_path = './data/'
    log_dir = 'output/logs/'

    initialize_seeds(args)
    begin = datetime.datetime.now()
    
    target_train_g, shadow_train_g, n_classes = prepare_graph_data(args)
    print_graph_statistics(target_train_g, shadow_train_g, n_classes, args)

    model, train_acc, train_auc, test_acc, test_auc, stat_dict = perform_attack(args, target_train_g, shadow_train_g)

    end = datetime.datetime.now()
    k = (end - begin).seconds
    
    save_results(args, stat_dict, log_dir, k, train_acc, train_auc, test_acc, test_auc)

if __name__ == '__main__':
    main()
