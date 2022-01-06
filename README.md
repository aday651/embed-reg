# README

Code for the paper [Asymptotics of L2 Regularized Network Embeddings](https://arxiv.org/abs/2201.01689).

## Requirements

Requires Stellargraph 1.2.1, Tensorflow 2.6.0, scikit-learm 0.24.1, tqdm, along
with any other packages required for the above three packages.

## Code

To run node classification or link prediction experiments, run 

```
python -m code.train_embed [[args]]
```

or 

```
python -m code.train_embed_link [[args]]
```

from the command line respectively, where `[[args]]` correspond to the
command line arguments
for each function. Note that the scripts expect to run from the parent
directory of the `code` folder; you will need to change the `import` 
statements in the associated python files if you move them around. The
`-h` command line argument will display the arguments (with descriptions)
of each of the two files.

### train_embed.py arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--dataset`|`Cora`|Dataset to perform training on. Available options: Cora,CiteSeer,PubMedDiabetes|
||`--emb-size`|`128`|Embedding dimension. Defaults to 128.|
||`--reg-weight`|`0.0`|Weight to use for L2 regularization. If norm_reg is True, then reg_weight/num_of_nodes is used instead.|
||`--norm-reg`||Boolean for whether to normalize the L2 regularization weight by the number of nodes in the graph. Defaults to false.|
||`--method`|`node2vec`|Algorithm to perform training on. Available options: node2vec,GraphSAGE,GCN,DGI|
||`--verbose`|`1`|Level of verbosity. Defaults to 1.|
||`--epochs`|`5`|Number of epochs through the dataset to be used for training.|
||`--optimizer`|`Adam`|Optimization algorithm to use for training.|
||`--learning-rate`|`0.001`|Learning rate to use for optimization.|
||`--batch-size`|`64`|Batch size used for training.|
||`--train-split`|`[0.01, 0.025, 0.05]`|Percentage(s) to use for the training split when using the learned embeddings for downstream classification tasks.|
||`--train-split-num`|`25`|Decides the number of random training/test splits to use for evaluating performance. Defaults to 50.|
||`--output-fname`|`None`|If not None, saves the hyperparameters and testing results to a .json file with filename given by the argument.|
||`--node2vec-p`|`1.0`|Hyperparameter governing probability of returning to source node.|
||`--node2vec-q`|`1.0`|Hyperparameter governing probability of moving to a node away from the source node.|
||`--node2vec-walk-number`|`50`|Number of walks used to generate a sample for node2vec.|
||`--node2vec-walk-length`|`5`|Walk length to use for node2vec.|
||`--dgi-sampler`|`fullbatch`|Specifies either a fullbatch or a minibatch sampling scheme for DGI.|
||`--gcn-activation`|`['relu']`|Determines the activations of each layer within a GCN. Defaults to a single layer with relu activation.|
||`--graphSAGE-aggregator`|`mean`|Specifies the aggreagtion rule used in GraphSAGE. Defaults to mean pooling.|
||`--graphSAGE-nbhd-sizes`|`[10, 5]`|Specify multiple neighbourhood sizes for sampling in GraphSAGE. Defaults to [10, 5].|
||`--tensorboard`||If toggles, saves Tensorboard logs for debugging purposes.|
||`--visualize-embeds`|`None`|If specified with a directory, saves an image of a TSNE 2D projection of the learned embeddings at the specified directory.|
||`--save-spectrum`|`None`|If specifies, saves the spectrum of the learned embeddings output by the algorithm.|

### train_embed_link.py arguments

|short|long|default|help|
| :--- | :--- | :--- | :--- |
|`-h`|`--help`||show this help message and exit|
||`--dataset`|`Cora`|Dataset to perform training on. Available options: Cora,CiteSeer,PubMedDiabetes|
||`--emb-size`|`128`|Embedding dimension. Defaults to 128.|
||`--reg-weight`|`0.0`|Weight to use for L2 regularization. If norm_reg is True, then reg_weight/num_of_nodes is used instead.|
||`--norm-reg`||Boolean for whether to normalize the L2 regularization weight by the number of nodes in the graph. Defaults to false.|
||`--method`|`node2vec`|Algorithm to perform training on. Available options: node2vec,GraphSAGE,GCN,DGI|
||`--verbose`|`1`|Level of verbosity. Defaults to 1.|
||`--epochs`|`5`|Number of epochs through the dataset to be used for training.|
||`--optimizer`|`Adam`|Optimization algorithm to use for training.|
||`--learning-rate`|`0.001`|Learning rate to use for optimization.|
||`--batch-size`|`64`|Batch size used for training.|
||`--test-split`|`0.1`|Split of edge/non-edge set to be used for testing.|
||`--output-fname`|`None`|If not None, saves the hyperparameters and testing results to a .json file with filename given by the argument.|
||`--node2vec-p`|`1.0`|Hyperparameter governing probability of returning to source node.|
||`--node2vec-q`|`1.0`|Hyperparameter governing probability of moving to a node away from the source node.|
||`--node2vec-walk-number`|`50`|Number of walks used to generate a sample for node2vec.|
||`--node2vec-walk-length`|`5`|Walk length to use for node2vec.|
||`--gcn-activation`|`['relu']`|Specifies layers in terms of their output activation (either relu or linear), with the number of arguments determining the length of the GCN. Defaults to a single layer with relu activation.|
||`--graphSAGE-aggregator`|`mean`|Specifies the aggreagtion rule used in GraphSAGE. Defaults to mean pooling.|
||`--graphSAGE-nbhd-sizes`|`[10, 5]`|Specify multiple neighbourhood sizes for sampling in GraphSAGE. Defaults to [25, 10].|
