# Data-free Model Extraction Attacks

This directory contains an implementation of data-free model extraction attacks on Graph Neural Networks (GNNs).

## Files

1. `example.py`: Example script demonstrating how to run data-free attacks
2. `models/`:
   - `generator.py`: Graph generator implementation
   - `victim.py`: Victim model implementations
3. `attacks/`:
   - `attack1.py`: Type I Attack implementation
   - `attack2.py`: Type II Attack implementation
   - `attack3.py`: Type III Attack implementation

## Running Data-free Attacks

The `example.py` script provides a complete example of how to run data-free attacks on GNN models. Here's how to use it:

```bash
python example.py
```

### Example Script Structure

The example script demonstrates:
1. Loading a dataset (Cora in this example)
2. Creating and training a victim model
3. Setting up attack parameters
4. Running a Type I attack

### Customizing Attack Parameters

The example shows the following default parameters which can be modified:
```python
# Attack parameters
noise_dim = 32
num_nodes = 500
num_queries = 300
generator_lr = 1e-6
surrogate_lr = 0.001
n_generator_steps = 2
n_surrogate_steps = 5
```

### Running Different Attack Types

The example includes code for all three attack types. To run different attacks, simply uncomment the desired attack in the `attacks` list:

```python
attacks = [
    ("Type I", TypeIAttack(generator, surrogate_model, victim_model, device, 
                           noise_dim, num_nodes, feature_dim, generator_lr, surrogate_lr,
                           n_generator_steps, n_surrogate_steps)),
    # Uncomment to run Type II attack
    # ("Type II", TypeIIAttack(...)),
    # Uncomment to run Type III attack
    # ("Type III", TypeIIIAttack(...))
]
```

The example script provides a template that can be easily modified to run attacks with different parameters or on different models. See the commented code in the script for Type II and Type III attack implementations.
