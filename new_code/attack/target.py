import os
import dgl
import time
import numpy as np
import torch as th
import torch.nn.functional as F
import torch.optim as optim
import torch.nn as nn
th.manual_seed(1)

from utils.load_model import get_gnn_model
from utils.model_evaluation import compute_accuracy, evaluate_model

def get_dataloader(train_graph, args):
    # Set up the data loader for training
    train_node_ids = th.tensor(range(0, len(train_graph.nodes())))
    sampler = dgl.dataloading.MultiLayerFullNeighborSampler(2)  # Sampling neighbors
    return dgl.dataloading.DataLoader(
        train_graph,
        train_node_ids,
        sampler,
        batch_size=args.batch_size,
        shuffle=True,
        drop_last=False,
        num_workers=args.num_workers
    )

def initialize_model_and_optimizer(args):
    # Initialize model, loss function, and optimizer
    model = get_gnn_model(args).to(args.device)
    loss_function = nn.CrossEntropyLoss().to(args.device)
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    return model, loss_function, optimizer

def training_step(model, blocks, batch_inputs, batch_labels, loss_function, optimizer, args, tic_step):
    # One training step: forward, loss calculation, and backpropagation
    blocks = [block.int().to(args.device) for block in blocks]
    batch_pred = model(blocks, batch_inputs)  # Model's prediction
    batch_pred = F.softmax(batch_pred, dim=1)  # Apply softmax for classification
    loss = loss_function(batch_pred, batch_labels)  # Compute loss
    optimizer.zero_grad()  # Zero out gradients
    loss.backward()  # Backpropagate
    optimizer.step()  # Update weights
    return loss, batch_pred

def log_info(epoch, step, loss, batch_pred, batch_labels, iter_tput):
    # Print training details like loss, accuracy, and throughput
    acc = compute_accuracy(batch_pred, batch_labels)
    print(f'Epoch {epoch:05d} | Step {step:05d} | Loss {loss.item():.4f} | '
          f'Train Acc {acc.item():.4f} | Speed (samples/sec) {np.mean(iter_tput[3:]):.4f}')

def evaluate_and_log(model, train_graph, test_graph, train_node_ids, test_node_ids, args):
    # Evaluate the model on train and test datasets
    train_acc, _ = evaluate_model(model, train_graph, train_graph.ndata['features'], 
                            train_graph.ndata['labels'], train_node_ids, args.device)
    print(f'Train Acc {train_acc:.4f}')
    
    test_acc, _ = evaluate_model(model, test_graph, test_graph.ndata['features'], 
                           test_graph.ndata['labels'], test_node_ids, args.device)
    print(f'Test Acc: {test_acc:.4f}')

def save_trained_model(model, args):
    # Save the trained model to disk
    model_save_path = os.path.join(
        args.model_save_path, 
        f'{args.setting}_{args.dataset}_{args.model}_{args.mode}'
        f'_{args.prop if args.prop else ""}.pth'
    )
    print(f"Training complete, model saved to {model_save_path}")
    th.save(model.state_dict(), model_save_path)

def run_gnn(args, data):
    train_graph, test_graph = data
    train_node_ids = th.tensor(range(0, len(train_graph.nodes())))
    test_node_ids = th.tensor(range(0, len(test_graph.nodes())))

    # Initialize DataLoader for training
    dataloader = get_dataloader(train_graph, args)

    # Initialize model, loss function, and optimizer
    model, loss_function, optimizer = initialize_model_and_optimizer(args)

    iter_tput = []  # List for tracking throughput during training
    avg_epoch_time = 0  # Average time taken per epoch

    # Training loop
    for epoch in range(args.num_epochs):
        epoch_start_time = time.time()

        # Loop through batches
        for step, (_, seeds, blocks) in enumerate(dataloader):
            batch_inputs = blocks[0].srcdata['features']
            batch_labels = blocks[-1].dstdata['labels'].to(device=args.device, dtype=th.long)

            # Perform a training step (forward, backward, and optimization)
            loss, batch_pred = training_step(model, blocks, batch_inputs, batch_labels, loss_function, optimizer, args, epoch_start_time)

            # Track throughput and log training info every few steps
            iter_tput.append(len(seeds) / (time.time() - epoch_start_time))
            if step % args.log_every == 0:
                log_info(epoch, step, loss, batch_pred, batch_labels, iter_tput)

            epoch_start_time = time.time()

        # Log time taken for each epoch
        epoch_end_time = time.time()
        print(f'Epoch {epoch}, Time(s): {epoch_end_time - epoch_start_time:.4f}')

        # Calculate average time after the first few epochs
        if epoch >= 5:
            avg_epoch_time += epoch_end_time - epoch_start_time

        # Evaluate the model periodically
        if epoch % args.eval_every == 0 and epoch != 0:
            evaluate_and_log(model, train_graph, test_graph, train_node_ids, test_node_ids, args)

    # Save the trained model
    save_trained_model(model, args)

    # Final evaluation
    evaluate_and_log(model, train_graph, test_graph, train_node_ids, test_node_ids, args)

    return evaluate_and_log(model, train_graph, test_graph, train_node_ids, test_node_ids, args)
