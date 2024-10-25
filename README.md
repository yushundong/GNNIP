# PyGIP Installation Guide

PyGIP supports multiple CUDA versions and provides two installation methods. Choose the method that best suits your needs.

## Method 1: Direct Installation

Create and activate a new conda environment:
```bash
conda create -n pygip python=3.10.14
conda activate pygip
```

### Choose your CUDA version:

#### For CUDA 11.x users:
```bash
pip install pygip -f https://data.dgl.ai/wheels/torch-2.3/cu118/repo.html --extra-index-url https://download.pytorch.org/whl/cu118
```

#### For CUDA 12.x users:
```bash
pip install pygip -f https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html --extra-index-url https://data.dgl.ai/wheels/torch-2.3/cu121/repo.html
```

## Method 2: Environment Setup 

This method uses a predefined environment.yml file and is recommended for development:

1. Create and activate the environment:
```bash
conda env create -f environment.yml -n pygip
conda activate pygip
```

2. Install DGL manually (required due to DGL 2.2.1 dependency issues):
```bash
pip install dgl -f https://data.dgl.ai/wheels/repo.html
```

3. Set up the Python path (run this from the PyGIP root directory):
```bash
# Linux/Mac:
export PYTHONPATH=`pwd`

# Windows:
set PYTHONPATH=%cd%
```

4. Test the installation:
```bash
python examples/examples.py
```

## Verifying CUDA Setup

To verify your CUDA installation is working correctly:
```python
import torch
print("CUDA Available:", torch.cuda.is_available())
print("CUDA Version:", torch.version.cuda)
print("GPU Device:", torch.cuda.get_device_name(0) if torch.cuda.is_available() else "No GPU found")
```

## Troubleshooting

If you encounter CUDA-related issues:

1. Ensure your NVIDIA drivers are up to date:
```bash
nvidia-smi
```

2. If you need to reinstall PyTorch with a specific CUDA version:
```bash
# Remove existing torch installation
pip uninstall torch torch-geometric -y

# For CUDA 11.x:
pip install torch --index-url https://download.pytorch.org/whl/cu118
pip install torch-geometric==2.5.0

# For CUDA 12.x:
pip install torch --index-url https://download.pytorch.org/whl/cu121
pip install torch-geometric==2.5.0
```

3. Verify DGL installation:
```bash
python -c "import dgl; print(dgl.__version__)"
```

## Requirements

PyGIP has been tested with the following core dependencies:
- Python 3.10.14
- PyTorch 2.3.0
- torch-geometric 2.5.0
- DGL 2.2.1

For a complete list of dependencies, see the `requirements.txt` file in the repository.y


# Attack

## Model Extraction Attacks against Graph Neural Network

### Example Python Code

```python

# Importing necessary classes and functions from the pygip library.
from pygip.datasets.datasets import *  # Import all available datasets.
from pygip.protect import *  # Import all core algorithms.

# Loading the Cora dataset, which is commonly used in graph neural network research.
dataset = Cora()

# Initializing a model extraction attack with the Cora dataset.
# The second parameter (0.25) might represent the fraction of the data.
modelExtractionAttack = ModelExtractionAttack0(dataset, 0.25)

# Executing the attack on the model.
modelExtractionAttack.attack()
```

### Example Usage

```bash
python3 examples/examples.py
```

### Attack 0

#### 1. Attack 0 on Cora

```
Enter dataset name (Cora, Citeseer, PubMed): Cora
Enter attack type (0-5): 0
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:01<00:00, 158.74it/s]
=========Model Extracting==========================
100%|█████████████████| 200/200 [00:02<00:00, 75.90it/s]
========================Final results:=========================================
Fidelity: 0.8567208271787297, Accuracy: 0.7853274249138356
```

#### 2. Attack 0 on Citeseer

```
Enter dataset name (Cora, Citeseer, PubMed): Citeseer
Enter attack type (0-5): 0
  NumNodes: 3327
  NumEdges: 9228
  NumFeats: 3703
  NumClasses: 6
  NumTrainingSamples: 120
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:02<00:00, 67.16it/s]
=========Model Extracting==========================
100%|█████████████████| 200/200 [00:06<00:00, 33.13it/s]
========================Final results:=========================================
Fidelity: 0.7784455128205128, Accuracy: 0.6778846153846154
```

#### . Attack 0 on DBLP 
```
Enter dataset name (Cora, Citeseer, PubMed or more): DBLP
Currently, only attack 0 is supported for this dataset.
Enter attack type (0-5): 0
=========Target Model Generating==========================
100%|█████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 281.01it/s]
/opt/homebrew/Caskroom/miniconda/base/envs/pygip/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
=========Model Extracting==========================
100%|█████████████████████████████████████████████████████████████████████| 200/200 [00:01<00:00, 135.48it/s]
========================Final results:=========================================
Fidelity: 0.2763720013144923, Accuracy: 0.29477489319750244
```

#### . Attack 0 on PubMed

```
Enter dataset name (Cora, Citeseer, PubMed): PubMed
Enter attack type (0-5): 0
  NumNodes: 19717
  NumEdges: 88651
  NumFeats: 500
  NumClasses: 3
  NumTrainingSamples: 60
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:04<00:00, 49.73it/s]
=========Model Extracting==========================
100%|█████████████████| 200/200 [00:11<00:00, 17.35it/s]
========================Final results:=========================================
Fidelity: 0.9076954287259941, Accuracy: 0.7790100081146876
```

#### . Attack 0 on Flickr (Stuck somewhere)

```
Enter dataset name (Cora, Citeseer, PubMed or more): Flickr
Currently, only attack 0 is supported for this dataset.
Enter attack type (0-5): 0
Downloading ./downloads/flickr.zip from https://data.dgl.ai/dataset/flickr.zip...
./downloads/flickr.zip: 100%|████████████████████████████████████████████████████████████████████████████████████████| 25.7M/25.7M [00:17<00:00, 1.44MB/s]
Extracting file to ./downloads/flickr_b05c56ca
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:25<00:00,  7.91it/s]
```

#### . Attack 0 on WiKi-CS (Stuck somewhere)

```
Enter dataset name (Cora, Citeseer, PubMed or more): WiKi-CS
Currently, only attack 0 is supported for this dataset.
Enter attack type (0-5): 0
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:08<00:00, 24.26it/s]
```

#### . Attack 0 on Facebook (Stuck somewhere)
```
Enter dataset name (Cora, Citeseer, PubMed or more): Facebook
Currently, only attack 0 is supported for this dataset.
Enter attack type (0-5): 0
Graph: Graph(num_nodes=22470, num_edges=342004,
      ndata_schemes={}
      edata_schemes={})
Graph nodes: 22470
Graph edges: 342004
Node feature shape: torch.Size([22470, 128])
Node label shape: torch.Size([22470])
Train mask shape: torch.Size([22470])
Test mask shape: torch.Size([22470])
=========Target Model Generating==========================
100%|████████████████████████| 200/200 [00:04<00:00, 44.61it/s]

```

#### . Attack 0 on Polblogs (Stuck somewhere)
```
Enter dataset name (Cora, Citeseer, PubMed or more): Ploblogs
Currently, only attack 0 is supported for this dataset.
Enter attack type (0-5): 0
Downloading https://netset.telecom-paris.fr/datasets/polblogs.tar.gz
Extracting downloads/raw/polblogs.tar.gz
=========Target Model Generating==========================
100%|█████████████████████████████████████████████████████████| 200/200 [00:04<00:00, 45.02it/s]

```

#### . Attack 0 on LastFM (Stuck somewhere)
```
Enter dataset name (Cora, Citeseer, PubMed or more): LastFM
Currently, only attack 0 is supported for this dataset.
Enter attack type (0-5): 0
Downloading https://graphmining.ai/datasets/ptg/lastfm_asia.npz
=========Target Model Generating==========================
100%|█████████████████████████████████████████████████████████| 200/200 [00:04<00:00, 45.51it/s]

```



#### . Attack 0 on Reddit (Out of Memory)

```
Enter attack type (0-5): 0
Downloading ./downloads/reddit.zip from https://data.dgl.ai/dataset/reddit.zip...
./downloads/reddit.zip: 100%|███████████████████████████████████████████████████████████████████████████████████| 1.40G/1.40G [03:21<00:00, 6.94MB/s]
Extracting file to ./downloads/reddit_69f818f5
 =========Target Model Generating==========================
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [29:23<00:00,  8.82s/it]
Killed: 9
(pygip) Kevins-MacBook-Pro:GNNIP kevinshuey$  /opt/homebrew/Caskroom/miniconda/base/envs/pygip/lib/python3.10/multiprocessing/resource_tracker.py:224: UserWarning: resource_tracker: There appear to be 1 leaked semaphor
```

#### . Attack 0 on PTC

```
Enter dataset name (Cora, Citeseer, PubMed or more): PTC
Currently, only attack 0 is supported for this dataset.
Enter attack type (0-5): 0
Downloading ./downloads/GINDataset.zip from https://raw.githubusercontent.com/weihua916/powerful-gnns/master/dataset.zip...
./downloads/GINDataset.zip: 100%|████████████████████████████████████████████████████████████████████████████████████| 33.4M/33.4M [00:27<00:00, 1.21MB/s]
Extracting file to ./downloads/GINDataset
=========Target Model Generating==========================
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 641.49it/s]
/opt/homebrew/Caskroom/miniconda/base/envs/pygip/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
=========Model Extracting==========================
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 287.41it/s]
========================Final results:=========================================
Fidelity: 0.9042553191489362, Accuracy: 0.8617021276595744
```

#### . Attack 0 on NCI1

```
Enter dataset name (Cora, Citeseer, PubMed or more): NCI1
Currently, only attack 0 is supported for this dataset.
Enter attack type (0-5): 0

=========Target Model Generating==========================
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 588.27it/s]
/opt/homebrew/Caskroom/miniconda/base/envs/pygip/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
=========Model Extracting==========================
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 235.10it/s]
========================Final results:=========================================
Fidelity: 0.951310861423221, Accuracy: 0.9138576779026217
```

#### . Attack 0 on PROTEINS

```
Enter dataset name (Cora, Citeseer, PubMed or more): PROTEINS
Currently, only attack 0 is supported for this dataset.
Enter attack type (0-5): 0
=========Target Model Generating==========================
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 604.61it/s]
/opt/homebrew/Caskroom/miniconda/base/envs/pygip/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
=========Model Extracting==========================
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 244.43it/s]
========================Final results:=========================================
Fidelity: 0.9914893617021276, Accuracy: 0.8822695035460993
```

#### . Attack 0 on IMDB-BINARY

```
Enter dataset name (Cora, Citeseer, PubMed or more): COLLAB
Currently, only attack 0 is supported for this dataset.
Enter attack type (0-5): 0
=========Target Model Generating==========================
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 485.52it/s]
/opt/homebrew/Caskroom/miniconda/base/envs/pygip/lib/python3.10/site-packages/dgl/heterograph.py:92: DGLWarning: Recommend creating graphs by `dgl.graph(data)` instead of `dgl.DGLGraph(data)`.
  dgl_warning(
=========Model Extracting==========================
100%|██████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:00<00:00, 214.98it/s]
========================Final results:=========================================
Fidelity: 1.0, Accuracy: 1.0
```

#### . Attack 0 on Computers (Stuck somewhere)

```
Enter dataset name (Cora, Citeseer, PubMed or more): Computers
Currently, only attack 0 is supported for this dataset.
Enter attack type (0-5): 0
Downloading ./downloads/amazon_co_buy_computer.zip from https://data.dgl.ai/dataset/amazon_co_buy_computer.zip...
./downloads/amazon_co_buy_computer.zip: 100%|████████████████████████████████████████████████████████████████████████| 3.42M/3.42M [00:01<00:00, 1.81MB/s]
Extracting file to ./downloads/amazon_co_buy_computer_b5999b2e
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:08<00:00, 22.39it/s]
```

#### . Attack 0 on Photos (Stuck somewhere)

```
Enter dataset name (Cora, Citeseer, PubMed or more): Photo
Currently, only attack 0 is supported for this dataset.
Enter attack type (0-5): 0
Downloading ./downloads/amazon_co_buy_photo.zip from https://data.dgl.ai/dataset/amazon_co_buy_photo.zip...
./downloads/amazon_co_buy_photo.zip: 100%|███████████████████████████████████████████████████████████████████████████| 1.78M/1.78M [00:01<00:00, 1.22MB/s]
Extracting file to ./downloads/amazon_co_buy_photo_b75d805d
=========Target Model Generating==========================
100%|███████████████████████████████████████████████████████████████████████████████████████████████████████████████████| 200/200 [00:04<00:00, 43.96it/s]

```

#### . Attack 0 on Yelp (Such a large dataset)

```
Enter dataset name (Cora, Citeseer, PubMed or more): Yelp    
Currently, only attack 0 is supported for this dataset.
Enter attack type (0-5): 0
Downloading ./downloads/yelp.zip from https://data.dgl.ai/dataset/yelp.zip...
./downloads/yelp.zip:  32%|████████████████████████████▋                                                              | 311M/987M [09:14<1:11:11, 158kB/s]
```

### Attack 1

#### 1. Attack 1 on Cora

```
Enter dataset name (Cora, Citeseer, PubMed): Cora
Enter attack type (0-5): 1
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:01<00:00, 146.82it/s]
Net_shadow(
  (layer1): GraphConv(in=1433, out=16, normalization=both, activation=None)
  (layer2): GraphConv(in=16, out=7, normalization=both, activation=None)
)
===================Model Extracting================================
100%|█████████████████| 200/200 [00:01<00:00, 100.62it/s]
Fidelity: 0.21762948207171315, Accuracy: 0.3341633466135458
```

#### 2. Attack 1 on Citeseer

```
Enter dataset name (Cora, Citeseer, PubMed): Citeseer
Enter attack type (0-5): 1
  NumNodes: 3327
  NumEdges: 9228
  NumFeats: 3703
  NumClasses: 6
  NumTrainingSamples: 120
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:03<00:00, 66.00it/s]
Net_shadow(
  (layer1): GraphConv(in=3703, out=16, normalization=both, activation=None)
  (layer2): GraphConv(in=16, out=6, normalization=both, activation=None)
)
===================Model Extracting================================
100%|█████████████████| 200/200 [00:04<00:00, 43.72it/s]
Fidelity: 0.6368481157213551, Accuracy: 0.6596878568709554
```

#### 3. Attack 1 on PubMed

```
Enter dataset name (Cora, Citeseer, PubMed): PubMed
Enter attack type (0-5): 1
  NumNodes: 19717
  NumEdges: 88651
  NumFeats: 500
  NumClasses: 3
  NumTrainingSamples: 60
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:04<00:00, 48.11it/s]
Net_shadow(
  (layer1): GraphConv(in=500, out=16, normalization=both, activation=None)
  (layer2): GraphConv(in=16, out=3, normalization=both, activation=None)
)
===================Model Extracting================================
100%|█████████████████| 200/200 [00:04<00:00, 44.45it/s]
Fidelity: 0.7714150496923805, Accuracy: 0.8506073513172425
```

### Attack 2

#### 1. Attack 2 on Cora

```
Enter dataset name (Cora, Citeseer, PubMed): Cora
Enter attack type (0-5): 2
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:01<00:00, 149.31it/s]
100%|█████████████████| 200/200 [00:04<00:00, 46.31it/s]
Fidelity: 0.791, Accuracy: 0.754
```

#### 2. Attack 2 on Citeseer

```
Enter dataset name (Cora, Citeseer, PubMed): Citeseer
Enter attack type (0-5): 2
  NumNodes: 3327
  NumEdges: 9228
  NumFeats: 3703
  NumClasses: 6
  NumTrainingSamples: 120
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:02<00:00, 69.09it/s]
100%|█████████████████| 200/200 [00:05<00:00, 34.15it/s]
Fidelity: 0.618, Accuracy: 0.521
```

#### 3. Attack 2 on PubMed

```
Enter dataset name (Cora, Citeseer, PubMed): PubMed
Enter attack type (0-5): 2
  NumNodes: 19717
  NumEdges: 88651
  NumFeats: 500
  NumClasses: 3
  NumTrainingSamples: 60
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:03<00:00, 51.98it/s]
100%|█████████████████| 200/200 [03:21<00:00,  1.01s/it]
Fidelity: 0.91, Accuracy: 0.782
```

### Attack 3

#### 1. Attack 3 on Cora

```
Enter dataset name (Cora, Citeseer, PubMed): Cora
Enter attack type (0-5): 3
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:01<00:00, 168.47it/s]
generated_train_mask 1977
  0%|                                                                                                                                                                                                                                                                                                                                          | 0/300 [00:00<?, ?it/s]/Users/haihaosun/Desktop/focusing/github/GNNIP/pygip/protect/gnn_mea.py:766: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/IndexingUtils.h:28.)
  loss_a = F.nll_loss(logp_a[generated_train_mask],
/Users/haihaosun/Desktop/focusing/github/GNNIP/pygip/protect/gnn_mea.py:767: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/IndexingUtils.h:28.)
  generated_labels[generated_train_mask])
/Users/haihaosun/anaconda3/envs/pygip/lib/python3.10/site-packages/torch/autograd/graph.py:744: UserWarning: indexing with dtype torch.uint8 is now deprecated, please use a dtype torch.bool instead. (Triggered internally at /Users/runner/work/pytorch/pytorch/pytorch/aten/src/ATen/native/IndexingUtils.h:28.)
  return Variable._execution_engine.run_backward(  # Calls into the C++ engine to run the backward pass
100%|█████████████████| 300/300 [00:02<00:00, 106.61it/s]
Fidelity: 0.7894321766561514, Accuracy: 0.8154574132492114
```

#### 2. Attack 3 on Citeseer (Original Implementation Failed.)

```
Enter dataset name (Cora, Citeseer, PubMed): Citeseer
Enter attack type (0-5): 3
  NumNodes: 3327
  NumEdges: 9228
  NumFeats: 3703
  NumClasses: 6
  NumTrainingSamples: 120
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:03<00:00, 66.66it/s]
generated_train_mask 1833
Traceback (most recent call last):
  File "/Users/haihaosun/Desktop/focusing/github/GNNIP/examples/examples.py", line 94, in <module>
    run_attack(attack_type, dataset_name)
  File "/Users/haihaosun/Desktop/focusing/github/GNNIP/examples/examples.py", line 85, in run_attack
    a.attack()
  File "/Users/haihaosun/Desktop/focusing/github/GNNIP/pygip/protect/gnn_mea.py", line 704, in attack
    generated_train_mask[i] = 0
IndexError: index 1833 is out of bounds for dimension 0 with size 1833

Original Implementation Failed.
```

#### 3. Attack 3 on PubMed (Missing taget_graph_index file.)

```
Enter dataset name (Cora, Citeseer, PubMed): PubMed
Enter attack type (0-5): 3
  NumNodes: 19717
  NumEdges: 88651
  NumFeats: 500
  NumClasses: 3
  NumTrainingSamples: 60
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:03<00:00, 52.31it/s]
Traceback (most recent call last):
  File "/Users/haihaosun/Desktop/focusing/github/GNNIP/examples/examples.py", line 94, in <module>
    run_attack(attack_type, dataset_name)
  File "/Users/haihaosun/Desktop/focusing/github/GNNIP/examples/examples.py", line 85, in run_attack
    a.attack()
  File "/Users/haihaosun/Desktop/focusing/github/GNNIP/pygip/protect/gnn_mea.py", line 618, in attack
    fileObject = open('./pygip/data/attack3_shadow_graph/' + self.dataset.dataset_name +
FileNotFoundError: [Errno 2] No such file or directory: './pygip/data/attack3_shadow_graph/pubmed/target_graph_index.txt'
```

### Attack 4

#### 1. Attack 4 on Cora

```
Enter dataset name (Cora, Citeseer, PubMed): Cora
Enter attack type (0-5): 4
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 300/300 [00:02<00:00, 111.63it/s]
Fidelity: 0.13808664259927797, Accuracy: 0.07581227436823104
```

#### 2. Attack 4 on Citeseer

```
Enter dataset name (Cora, Citeseer, PubMed): Citeseer
Enter attack type (0-5): 4
  NumNodes: 3327
  NumEdges: 9228
  NumFeats: 3703
  NumClasses: 6
  NumTrainingSamples: 120
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 300/300 [00:05<00:00, 54.71it/s]
Fidelity: 0.2325925925925926, Accuracy: 0.2069135802469136
```

#### 3. Attack 4 on PubMed (Missing taget_graph_index file.)

```
Enter dataset name (Cora, Citeseer, PubMed): PubMed
Enter attack type (0-5): 4
  NumNodes: 19717
  NumEdges: 88651
  NumFeats: 500
  NumClasses: 3
  NumTrainingSamples: 60
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
Traceback (most recent call last):
  File "/Users/haihaosun/Desktop/focusing/github/GNNIP/examples/examples.py", line 94, in <module>
    run_attack(attack_type, dataset_name)
  File "/Users/haihaosun/Desktop/focusing/github/GNNIP/examples/examples.py", line 85, in run_attack
    a.attack()
  File "/Users/haihaosun/Desktop/focusing/github/GNNIP/pygip/protect/gnn_mea.py", line 800, in attack
    fileObject = open('./pygip/data/' + self.dataset.dataset_name +
FileNotFoundError: [Errno 2] No such file or directory: './pygip/data/pubmed/target_graph_index.txt'
```

### Attack 5

#### 1. Attack 5 on Cora

```
Enter dataset name (Cora, Citeseer, PubMed): Cora
Enter attack type (0-5): 5
  NumNodes: 2708
  NumEdges: 10556
  NumFeats: 1433
  NumClasses: 7
  NumTrainingSamples: 140
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 300/300 [00:02<00:00, 112.59it/s]
Fidelity: 0.010830324909747292, Accuracy: 0.15433212996389892
```

#### 2. Attack 5 on Citeseer

```
Enter dataset name (Cora, Citeseer, PubMed): Citeseer
Enter attack type (0-5): 5
  NumNodes: 3327
  NumEdges: 9228
  NumFeats: 3703
  NumClasses: 6
  NumTrainingSamples: 120
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 300/300 [00:05<00:00, 54.44it/s]
Fidelity: 0.2325925925925926, Accuracy: 0.2069135802469136
```

#### 3. Attack 5 on PubMed (Missing taget_graph_index file.)

```
Enter dataset name (Cora, Citeseer, PubMed): PubMed
Enter attack type (0-5): 5
  NumNodes: 19717
  NumEdges: 88651
  NumFeats: 500
  NumClasses: 3
  NumTrainingSamples: 60
  NumValidationSamples: 500
  NumTestSamples: 1000
Done loading data from cached files.
Traceback (most recent call last):
  File "/Users/haihaosun/Desktop/focusing/github/GNNIP/examples/examples.py", line 94, in <module>
    run_attack(attack_type, dataset_name)
  File "/Users/haihaosun/Desktop/focusing/github/GNNIP/examples/examples.py", line 85, in run_attack
    a.attack()
  File "/Users/haihaosun/Desktop/focusing/github/GNNIP/pygip/protect/gnn_mea.py", line 997, in attack
    fileObject = open('./pygip/data/' + self.dataset.dataset_name +
FileNotFoundError: [Errno 2] No such file or directory: './pygip/data/pubmed/target_graph_index.txt'
```

# Defense

## Watermarking Graph Neural Networks By Random Graphs

### Example Python Code

```python
# Importing necessary classes and functions from the pygip library.
from pygip.protect.Defense import Watermark_sage
from pygip.protect import *

# Use dataset with a watermark graph to train a target model
model = Watermark_sage(Cora(),0.25)
# Choose Cora as dataset. The second parameter represents attack model and 1 is ModelExtractionAttack0. The third parameter represents dataset and 1 is Cora.
# Use the target model to defense against model extraction attack and use the accuracy of the prediction of labels to distinguish whether the model is trained independently
model.watermark_attack(Cora(), 1, 1)
```

### Example Usage

```
python3 examples/Watermarking_Graph_Neural_Networks_By_Random_Graphs.py
```

### Attack Model: Model Extraction Attacks against Graph Neural Network

#### Attack0-Watermark

##### 1. Attack0-Watermark on Cora

Follow the instructions to enter 1 and 1 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
1
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
1
```

We present the sample log as follows

```
NumNodes: 2708
NumEdges: 10556
NumFeats: 1433
NumClasses: 7
NumTrainingSamples: 140
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:07<00:00, 6.53it/s]
Marked Acc: 0.7890
100%|██████████████████| 15/15 [00:00<00:00, 145.31it/s]
Final results
Non-Marked Acc: 0.1400, Marked Acc: 0.4980, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:03<00:00, 61.88it/s]
=========Model Extracting==========================
100%|█████████████████| 200/200 [00:05<00:00, 38.38it/s]
========================Final results:=========================================
Fidelity: 0.8598356694055099, Accuracy: 0.7878202029966167
Watermark Graph - Accuracy: 0.24
```

##### 2. Attack0-Watermark on Citeseer

Follow the instructions to enter 1 and 2 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
1
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
2
```

We present the sample log as follows

```
NumNodes: 3327
NumEdges: 9228
NumFeats: 3703
NumClasses: 6
NumTrainingSamples: 120
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:09<00:00, 5.03it/s]
Marked Acc: 0.7080
100%|█████████████████| 15/15 [00:00<00:00, 96.68it/s]
Final results
Non-Marked Acc: 0.1800, Marked Acc: 0.4860, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:11<00:00, 18.11it/s]
=========Model Extracting==========================
100%|█████████████████| 200/200 [00:17<00:00, 11.31it/s]
========================Final results:=========================================
Fidelity: 0.8484011054086064, Accuracy: 0.7090406632451638
Watermark Graph - Accuracy: 0.06
```

##### 3. Attack0-Watermark on PubMed

Follow the instructions to enter 1 and 3 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
1
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
3
```

We present the sample log as follows

```
NumNodes: 19717
NumEdges: 88651
NumFeats: 500
NumClasses: 3
NumTrainingSamples: 60
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:06<00:00, 7.49it/s]
Marked Acc: 0.7800
100%|██████████████████| 15/15 [00:00<00:00, 136.56it/s]
Final results
Non-Marked Acc: 0.2400, Marked Acc: 0.5640, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:07<00:00, 26.09it/s]
=========Model Extracting==========================
100%|█████████████████| 200/200 [00:14<00:00, 13.71it/s]
========================Final results:=========================================
Fidelity: 0.9202077431539188, Accuracy: 0.7921219479293133
Watermark Graph - Accuracy: 0.32
```

#### Attack1-Watermark

##### 1. Attack1-Watermark on Cora

Follow the instructions to enter 2 and 1 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
2
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
1
```

We present the sample log as follows

```
NumNodes: 2708
NumEdges: 10556
NumFeats: 1433
NumClasses: 7
NumTrainingSamples: 140
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:07<00:00, 7.13it/s]
Marked Acc: 0.7740
100%|██████████████████| 15/15 [00:00<00:00, 150.18it/s]
Final results
Non-Marked Acc: 0.2000, Marked Acc: 0.5910, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:03<00:00, 56.25it/s]
Net_shadow(
(layer1): GraphConv(in=1433, out=16, normalization=both, activation=None)
(layer2): GraphConv(in=16, out=7, normalization=both, activation=None)
)
===================Model Extracting================================
100%|█████████████████| 200/200 [00:02<00:00, 66.87it/s]
Fidelity: 0.7750242954324587, Accuracy: 0.7769679300291545
Watermark Graph - Accuracy: 0.12
```

##### 2. Attack1-Watermark on Citeseer

Follow the instructions to enter 2 and 2 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
2
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
2
```

We present the sample log as follows

```
NumNodes: 3327
NumEdges: 9228
NumFeats: 3703
NumClasses: 6
NumTrainingSamples: 120
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:09<00:00, 5.20it/s]
Marked Acc: 0.7180
100%|█████████████████| 15/15 [00:00<00:00, 89.53it/s]
Final results
Non-Marked Acc: 0.1800, Marked Acc: 0.4220, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:11<00:00, 18.16it/s]
Net_shadow(
(layer1): GraphConv(in=3703, out=16, normalization=both, activation=None)
(layer2): GraphConv(in=16, out=6, normalization=both, activation=None)
)
===================Model Extracting================================
100%|█████████████████| 200/200 [00:07<00:00, 26.10it/s]
Fidelity: 0.6955547254389242, Accuracy: 0.6936869630183041
Watermark Graph - Accuracy: 0.18
```

##### 3. Attack1-Watermark on PubMed

Follow the instructions to enter 2 and 3 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
2
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
3
```

We present the sample log as follows

```
NumNodes: 19717
NumEdges: 88651
NumFeats: 500
NumClasses: 3
NumTrainingSamples: 60
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:05<00:00, 8.43it/s]
Marked Acc: 0.7660
100%|██████████████████| 15/15 [00:00<00:00, 196.00it/s]
Final results
Non-Marked Acc: 0.3000, Marked Acc: 0.7190, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:07<00:00, 25.77it/s]
Net_shadow(
(layer1): GraphConv(in=500, out=16, normalization=both, activation=None)
(layer2): GraphConv(in=16, out=3, normalization=both, activation=None)
)
===================Model Extracting================================
100%|█████████████████| 200/200 [00:07<00:00, 25.27it/s]
Fidelity: 0.813971783710075, Accuracy: 0.8138668904389783
Watermark Graph - Accuracy: 0.44
```

#### Attack2-Watermark

##### 1. Attack2-Watermark on Cora

Follow the instructions to enter 3 and 1 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
3
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
1
```

We present the sample log as follows

```
NumNodes: 2708
NumEdges: 10556
NumFeats: 1433
NumClasses: 7
NumTrainingSamples: 140
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:07<00:00, 6.48it/s]
Marked Acc: 0.7900
100%|██████████████████| 15/15 [00:00<00:00, 164.84it/s]
Final results
Non-Marked Acc: 0.1400, Marked Acc: 0.3080, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:03<00:00, 58.55it/s]
100%|█████████████████| 200/200 [00:09<00:00, 20.58it/s]
Fidelity: 0.793, Accuracy: 0.758
Watermark Graph - Accuracy: 0.1
```

##### 2. Attack2-Watermark on Citeseer

Follow the instructions to enter 3 and 2 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
3
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
2
```

We present the sample log as follows

```
NumNodes: 3327
NumEdges: 9228
NumFeats: 3703
NumClasses: 6
NumTrainingSamples: 120
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:09<00:00, 5.34it/s]
Marked Acc: 0.6940
100%|██████████████████| 15/15 [00:00<00:00, 107.36it/s]
Final results
Non-Marked Acc: 0.1200, Marked Acc: 0.3230, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:11<00:00, 17.95it/s]
100%|█████████████████| 200/200 [00:15<00:00, 13.08it/s]
Fidelity: 0.698, Accuracy: 0.587
Watermark Graph - Accuracy: 0.22
```

##### 3. Attack3-Watermark on PubMed

Follow the instructions to enter 3 and 3 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
3
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
3
```

We present the sample log as follows

```
NumNodes: 19717
NumEdges: 88651
NumFeats: 500
NumClasses: 3
NumTrainingSamples: 60
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:05<00:00, 8.83it/s]
Marked Acc: 0.7730
100%|██████████████████| 15/15 [00:00<00:00, 203.23it/s]
Final results
Non-Marked Acc: 0.3600, Marked Acc: 0.7300, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:07<00:00, 26.79it/s]
100%|█████████████████| 200/200 [3:03:21<00:00, 55.01s/it]
Fidelity: 0.887, Accuracy: 0.782
Watermark Graph - Accuracy: 0.34
```

#### Attack3-Watermark

##### 1. Attack3-Watermark on Cora

Follow the instructions to enter 4 and 1 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
4
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
1
```

We present the sample log as follows

```
NumNodes: 2708
NumEdges: 10556
NumFeats: 1433
NumClasses: 7
NumTrainingSamples: 140
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:08<00:00, 5.83it/s]
Marked Acc: 0.7840
100%|██████████████████| 15/15 [00:00<00:00, 137.76it/s]
Final results
Non-Marked Acc: 0.1600, Marked Acc: 0.4780, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:03<00:00, 51.53it/s]
generated_train_mask 1989
100%|█████████████████| 300/300 [00:05<00:00, 51.62it/s]
Fidelity: 0.7689274447949527, Accuracy: 0.8280757097791798
Watermark Graph - Accuracy: 0.14
```

##### 2. Attack3-Watermark on Citeseer

Follow the instructions to enter 4 and 2 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
4
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
2
```

We present the sample log as follows

```
NumNodes: 3327
NumEdges: 9228
NumFeats: 3703
NumClasses: 6
NumTrainingSamples: 120
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:10<00:00, 4.67it/s]
Marked Acc: 0.7000
100%|█████████████████| 15/15 [00:00<00:00, 95.75it/s]
Final results
Non-Marked Acc: 0.1400, Marked Acc: 0.3450, Watermark Acc: 0.9600
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:11<00:00, 17.85it/s]
generated_train_mask 1846
Traceback (most recent call last):
File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/protect/Defense.py", line 444, in <module>
defense.watermark_attack(Citeseer(), attack_name, dataset_name)
File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/protect/Defense.py", line 268, in watermark_attack
attack.attack()
File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/protect/gnn_mea.py", line 704, in attack
generated_train_mask[i] = 0
IndexError: index 1846 is out of bounds for dimension 0 with size 1846
```

##### 3. Attack3-Watermark on PubMed

Follow the instructions to enter 4 and 3 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
4
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
3
```

We present the sample log as follows

```
NumNodes: 19717
NumEdges: 88651
NumFeats: 500
NumClasses: 3
NumTrainingSamples: 60
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:06<00:00, 8.03it/s]
Marked Acc: 0.7750
100%|██████████████████| 15/15 [00:00<00:00, 202.34it/s]
Final results
Non-Marked Acc: 0.4200, Marked Acc: 0.6750, Watermark Acc: 1.0000
=========Target Model Generating==========================
100%|█████████████████| 200/200 [00:07<00:00, 26.74it/s]
Traceback (most recent call last):
File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/protect/Defense.py", line 447, in <module>
defense.watermark_attack(PubMed(), attack_name, dataset_name)
File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/protect/Defense.py", line 302, in watermark_attack
attack.attack()
File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/protect/gnn_mea.py", line 618, in attack
fileObject = open('./pygip/data/attack3_shadow_graph/' + self.dataset.dataset_name +
FileNotFoundError: [Errno 2] No such file or directory: './pygip/data/attack3_shadow_graph/pubmed/target_graph_index.txt'
```

#### Attack4-Watermark

##### 1. Attack4-Watermark on Cora

Follow the instructions to enter 5 and 1 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
5
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
1
```

We present the sample log as follows

```
NumNodes: 2708
NumEdges: 10556
NumFeats: 1433
NumClasses: 7
NumTrainingSamples: 140
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:08<00:00, 5.98it/s]
Marked Acc: 0.7890
100%|██████████████████| 15/15 [00:00<00:00, 166.12it/s]
Final results
Non-Marked Acc: 0.2000, Marked Acc: 0.3170, Watermark Acc: 1.0000
1408 1408
100%|█████████████████| 300/300 [00:05<00:00, 59.47it/s]
Fidelity: 0.010830324909747292, Accuracy: 0.15433212996389892
Watermark Graph - Accuracy: 0.14
```

##### 2. Attack4-Watermark on Citeseer

Follow the instructions to enter 5 and 2 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
5
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
2
```

We present the sample log as follows

```
NumNodes: 3327
NumEdges: 9228
NumFeats: 3703
NumClasses: 6
NumTrainingSamples: 120
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:10<00:00, 4.58it/s]
Marked Acc: 0.7070
100%|█████████████████| 15/15 [00:00<00:00, 97.85it/s]
Final results
Non-Marked Acc: 0.2400, Marked Acc: 0.1940, Watermark Acc: 1.0000
2325 2325
100%|█████████████████| 300/300 [00:10<00:00, 29.91it/s]
Fidelity: 0.08691358024691358, Accuracy: 0.12493827160493827
Watermark Graph - Accuracy: 0.2
```

##### 3. Attack4-Watermark on PubMed

Follow the instructions to enter 5 and 3 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
5
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
3
```

We present the sample log as follows

```
NumNodes: 19717
NumEdges: 88651
NumFeats: 500
NumClasses: 3
NumTrainingSamples: 60
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:06<00:00, 8.02it/s]
Marked Acc: 0.7730
100%|██████████████████| 15/15 [00:00<00:00, 182.80it/s]
Final results
Non-Marked Acc: 0.3600, Marked Acc: 0.5980, Watermark Acc: 1.0000
Traceback (most recent call last):
File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/protect/Defense.py", line 447, in <module>
defense.watermark_attack(PubMed(), attack_name, dataset_name)
File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/protect/Defense.py", line 307, in watermark_attack
attack.attack()
File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/protect/gnn_mea.py", line 800, in attack
fileObject = open('./pygip/data/' + self.dataset.dataset_name +
FileNotFoundError: [Errno 2] No such file or directory: './pygip/data/pubmed/target_graph_index.txt'
```

#### Attack5-Watermark

##### 1. Attack5-Watermark on Cora

Follow the instructions to enter 6 and 1 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
6
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
1
```

We present the sample log as follows

```
NumNodes: 2708
NumEdges: 10556
NumFeats: 1433
NumClasses: 7
NumTrainingSamples: 140
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:08<00:00, 6.11it/s]
Marked Acc: 0.7770
100%|██████████████████| 15/15 [00:00<00:00, 131.62it/s]
Final results
Non-Marked Acc: 0.1800, Marked Acc: 0.4630, Watermark Acc: 1.0000
100%|█████████████████| 300/300 [00:04<00:00, 61.87it/s]
Fidelity: 0.20126353790613719, Accuracy: 0.13357400722021662
Watermark Graph - Accuracy: 0.18
```

##### 2. Attack5-Watermark on Citeseer

Follow the instructions to enter 6 and 2 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
6
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
2
```

We present the sample log as follows

```
NumNodes: 3327
NumEdges: 9228
NumFeats: 3703
NumClasses: 6
NumTrainingSamples: 120
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:11<00:00, 4.40it/s]
Marked Acc: 0.6990
100%|██████████████████| 15/15 [00:00<00:00, 106.87it/s]
Final results
Non-Marked Acc: 0.1800, Marked Acc: 0.4000, Watermark Acc: 1.0000
100%|█████████████████| 300/300 [00:10<00:00, 28.05it/s]
Fidelity: 0.15604938271604937, Accuracy: 0.19358024691358025
Watermark Graph - Accuracy: 0.14
```

##### 3. Attack5-Watermark on PubMed

Follow the instructions to enter 6 and 3 in sequence

```
Please choose the number:
1.ModelExtractionAttack0
2.ModelExtractionAttack1
3.ModelExtractionAttack2
4.ModelExtractionAttack3
5.ModelExtractionAttack4
6.ModelExtractionAttack5
6
Please choose the number:
1.Cora
2.Citeseer
3.PubMed
3
```

We present the sample log as follows

```
NumNodes: 19717
NumEdges: 88651
NumFeats: 500
NumClasses: 3
NumTrainingSamples: 60
NumValidationSamples: 500
NumTestSamples: 1000
Done loading data from cached files.
100%|█████████████████| 50/50 [00:06<00:00, 7.95it/s]
Marked Acc: 0.7730
100%|██████████████████| 15/15 [00:00<00:00, 176.38it/s]
Final results
Non-Marked Acc: 0.3200, Marked Acc: 0.7020, Watermark Acc: 1.0000
Traceback (most recent call last):
File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/protect/Defense.py", line 447, in <module>
defense.watermark_attack(PubMed(), attack_name, dataset_name)
File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/protect/Defense.py", line 313, in watermark_attack
attack.attack()
File "/mnt/g/Final_Edition/GNNIP-main/GNNIP/protect/gnn_mea.py", line 997, in attack
fileObject = open('./pygip/data/' + self.dataset.dataset_name +
FileNotFoundError: [Errno 2] No such file or directory: './pygip/data/pubmed/target_graph_index.txt'
```
