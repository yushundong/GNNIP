import os
import numpy as np
import torch as th
import torch.nn as nn
import torch.nn.functional as F
from torch.optim.lr_scheduler import CosineAnnealingLR
from sklearn.metrics import roc_auc_score
th.manual_seed(0)

from model.mlp import MLP_ATTACK, MLP_ATTACK_PLUS, MLP_ATTACK_PLUS2, MLP_ATTACK_ALL

def _weights_init_normal(m):
    classname = m.__class__.__name__
    if classname.find('Linear') != -1:
        y = m.in_features
        m.weight.data.normal_(0.0, 1/np.sqrt(y))
        m.bias.data.fill_(0)

def save_attack_model(args, model):
    if not os.path.exists(args.attack_model_save_path):
        os.makedirs(args.attack_model_save_path)
    save_name = os.path.join(args.attack_model_save_path, f'{args.attack_model_prefix}_{args.dataset}_{args.target_model}_{args.shadow_model}_{args.node_topology}_{args.feature}_{args.edge_feature}.pth')
    th.save(model.state_dict(), save_name)
    print(f"Finish training, save model to {save_name}")

def load_attack_model(model, model_path, device):
    print("load model from:", model_path)
    state_dict = th.load(model_path, map_location=device)
    model.load_state_dict(state_dict)
    return model

def test_features(args, epoch, model, test_dataloader, num_features, stat_dict=None):
    device = args.device
    test_acc, correct, total, scores, targets = 0.0, 0, 0, [], []
    stat_dict = stat_dict or {}
    model.eval()

    with th.no_grad():
        for data in test_dataloader:
            inputs = [x.to(device) for x in data[:-1]]
            label = data[-1].to(device)

            outputs = model(*inputs)
            posteriors = F.softmax(outputs, dim=1)
            _, predicted = posteriors.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()

            if epoch == args.num_epochs - 1 and not args.diff:
                for i, posterior in zip(data[0], posteriors):
                    stat_dict[tuple(i.cpu().numpy())][f'{args.method}_attack_posterior'] = posterior.cpu().numpy()

            targets.extend(label.cpu().numpy().tolist())
            scores.extend([i.cpu().numpy()[1] for i in posteriors])

        test_acc = correct / total
        test_auc = roc_auc_score(targets, scores)
        print(f'Test Acc: {100. * test_acc:.3f}% ({correct}/{total}) AUC Score: {test_auc:.3f}')

    return test_acc, test_auc, stat_dict

def run_attack(args, in_dim, train_dataloader, test_dataloader, stat_dict, model_class, model_args=()):
    epoch, device = args.num_epochs, args.device
    model = model_class(*model_args).to(device)
    model.apply(_weights_init_normal)
    loss_fcn = nn.CrossEntropyLoss().to(device)
    optimizer = th.optim.Adam(model.parameters(), lr=args.lr) if args.optim == 'adam' else th.optim.SGD(model.parameters(), lr=args.lr)
    scheduler = CosineAnnealingLR(optimizer, T_max=epoch, eta_min=0)
    train_acc = 0.0

    for e in range(epoch):
        correct, total, targets, scores = 0, 0, [], []
        model.train()

        for _, *features, label in train_dataloader:
            optimizer.zero_grad()
            features = [f.to(device) for f in features]
            label = label.to(device)

            outputs = model(*features)
            posteriors = F.softmax(outputs, dim=1)
            loss = loss_fcn(posteriors, label)
            loss.backward()
            optimizer.step()

            _, predicted = posteriors.max(1)
            total += label.size(0)
            correct += predicted.eq(label).sum().item()
            targets.extend(label.cpu().detach().numpy().tolist())
            scores.extend([i.cpu().detach().numpy()[1] for i in posteriors])

        if args.scheduler:
            scheduler.step()

        train_acc = correct / total
        train_auc = roc_auc_score(targets, scores)
        print(f'[Epoch {e}] Train Acc: {100. * train_acc:.3f}% ({correct}/{total}) AUC Score: {train_auc:.3f}')

        if e == epoch - 1:
            test_acc, test_auc, stat_dict = test_features(args, e, model, test_dataloader, len(model_args), stat_dict)
            save_attack_model(args, model)
        else:
            test_acc, test_auc, _ = test_features(args, e, model, test_dataloader, len(model_args))

    return model, train_acc, train_auc, test_acc, test_auc, stat_dict

# Example for running attack with different feature combinations
def run_attack_one_feature(args, in_dim, train_dataloader, test_dataloader, stat_dict):
    return run_attack(args, in_dim, train_dataloader, test_dataloader, stat_dict, MLP_ATTACK)

def run_attack_two_features(args, posterior_feature_dim, train_dataloader, test_dataloader, stat_dict):
    model_class = MLP_ATTACK_PLUS2 if args.feature == 'posteriors_graph' or args.feature == 'label_graph' else MLP_ATTACK_PLUS
    return run_attack(args, posterior_feature_dim, train_dataloader, test_dataloader, stat_dict, model_class, (args.graph_feature_dim, posterior_feature_dim) if model_class == MLP_ATTACK_PLUS2 else (args.node_feature_dim, posterior_feature_dim))

def run_attack_three_features(args, posterior_feature_dim, train_dataloader, test_dataloader, stat_dict):
    return run_attack(args, posterior_feature_dim, train_dataloader, test_dataloader, stat_dict, MLP_ATTACK_ALL, (args.node_feature_dim, posterior_feature_dim, args.graph_feature_dim))
