# IDDG

Codes and datasets for paper [Invariant Learning for Domain Generalization on Graphs]
This work focuses on OOD problem on graph data, especially node-level prediction tasks and proposes a new approach Invariant-Domain-Generalization-on-Graphs for it. 


## Dependency

PYTHON 3.8, PyTorch 1.9.0, PyTorch Geometric 1.7.2

## Datasets

In our experiment, we consider three types of distribution shifts with three real-world datasets.

You can make a directory `./data` and download all the datasets through the Google drive:

      https://drive.google.com/drive/folders/15YgnsfSV_vHYTXe7I4e_hhGMcx0gKrO8?usp=sharing

Here is a brief introduction for three distribution shifts and the datasets:


- We use Twitch-Explicit and Facebook-100 datasets. These two datasets both contain multiple graphs. We use different graphs for training/validation/testing. The original Twitch dataset is from [Non-Homophily Benchmark](https://github.com/CUAI/Non-Homophily-Benchmarks/tree/main/data/twitch). For Facebook dataset, we use partial graphs for experiments, and its complete version could be obtained from [Facebook dataset](https://archive.org/details/oxford-2005-facebook-matrix).
- We use WebKB datasets. The original WebKB dataset is from (https://pytorch-geometric.readthedocs.io/en/latest/generated/torch_geometric.datasets.WebKB.html#torch_geometric.datasets.WebKB)
## Running the code

We do not provide the trained model since the training cost for each experiment is acceptable. To run the code, please refer to the bash script `run.sh`.


