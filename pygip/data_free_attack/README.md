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

The script will display the progress of the victim model training and the final accuracy of both the victim and surrogate models.
