# Data-free Model Extraction Attacks

This directory contains an implementation of data-free model extraction attacks on Graph Neural Networks (GNNs).

## Files

1. `example.py`: Interactive script demonstrating how to run data-free attacks
2. `models/`:
   - `generator.py`: Graph generator implementation
   - `victim.py`: Victim model implementations
3. `attacks/`:
   - `attack1.py`: Type I Attack implementation
   - `attack2.py`: Type II Attack implementation
   - `attack3.py`: Type III Attack implementation

## Running Data-free Attacks

The `example.py` script provides an interactive way to run data-free attacks on GNN models. Here's how to use it:

```bash
python example.py
```

When you run the script, it will:
1. Load the Cora dataset
2. Create and train a victim model
3. Prompt you to choose an attack type:
   ```
   Choose attack type (1, 2, or 3): 
   ```
4. Run the selected attack with the following default parameters:
   ```python
   noise_dim = 32
   num_nodes = 500
   num_queries = 300
   generator_lr = 1e-6
   surrogate_lr = 0.001
   n_generator_steps = 2
   n_surrogate_steps = 5
   ```

### Attack Types

1. Type I Attack: Basic model extraction attack
2. Type II Attack: Enhanced extraction with improved query strategy
3. Type III Attack: Advanced extraction with additional model architecture considerations

Choose the attack type by entering the corresponding number (1, 2, or 3) when prompted.

### Sample Output

```
Epoch 10/200, Train Loss: 1.7342, Val Loss: 1.8183, Val Acc: 0.7460
Epoch 20/200, Train Loss: 1.3186, Val Loss: 1.5902, Val Acc: 0.7860
Epoch 30/200, Train Loss: 0.8908, Val Loss: 1.3175, Val Acc: 0.7880
Epoch 40/200, Train Loss: 0.5930, Val Loss: 1.0948, Val Acc: 0.7860
Epoch 50/200, Train Loss: 0.4184, Val Loss: 0.9633, Val Acc: 0.7940
Epoch 60/200, Train Loss: 0.3414, Val Loss: 0.8969, Val Acc: 0.7900
Epoch 70/200, Train Loss: 0.2943, Val Loss: 0.8568, Val Acc: 0.7900
Epoch 80/200, Train Loss: 0.2577, Val Loss: 0.8343, Val Acc: 0.7940
Epoch 90/200, Train Loss: 0.2487, Val Loss: 0.8058, Val Acc: 0.7960
Epoch 100/200, Train Loss: 0.2310, Val Loss: 0.7731, Val Acc: 0.7880
Epoch 110/200, Train Loss: 0.2129, Val Loss: 0.7825, Val Acc: 0.7900
Epoch 120/200, Train Loss: 0.2092, Val Loss: 0.7696, Val Acc: 0.7920
Epoch 130/200, Train Loss: 0.1865, Val Loss: 0.7548, Val Acc: 0.7940
Epoch 140/200, Train Loss: 0.1748, Val Loss: 0.7522, Val Acc: 0.7960
Epoch 150/200, Train Loss: 0.1769, Val Loss: 0.7385, Val Acc: 0.7940
Epoch 160/200, Train Loss: 0.1682, Val Loss: 0.7552, Val Acc: 0.7920
Epoch 170/200, Train Loss: 0.1557, Val Loss: 0.7254, Val Acc: 0.7880
Epoch 180/200, Train Loss: 0.1608, Val Loss: 0.7346, Val Acc: 0.7940
Epoch 190/200, Train Loss: 0.1517, Val Loss: 0.7433, Val Acc: 0.7860
Epoch 200/200, Train Loss: 0.1482, Val Loss: 0.7290, Val Acc: 0.7940
Victim Model Accuracy: 0.8070

Choose attack type (1, 2, or 3): 2

Running Type II Attack...
Attacking: 100%|██████████████████████████████| 300/300 [01:09<00:00, 4.29it/s, Gen Loss=-0.3422, Surr Loss=0.4532] 
Type II Attack - Surrogate Model Accuracy: 0.8090
```

The script will display:
1. Training progress of the victim model, showing loss and validation accuracy
2. Final victim model accuracy
3. Progress bar during the attack
4. Final surrogate model accuracy
