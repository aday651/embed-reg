import numpy as np
import argparse
import json
import scipy
import matplotlib.pyplot as plt
import tensorflow as tf
from tensorflow import keras 
from tqdm import tqdm

from stellargraph.data import BiasedRandomWalk, UnsupervisedSampler 
from stellargraph.mapper import (
    Node2VecLinkGenerator, 
    Node2VecNodeGenerator, 
    GraphSAGENodeGenerator, 
    GraphSAGELinkGenerator, 
    CorruptedGenerator, 
    FullBatchNodeGenerator,
    FullBatchLinkGenerator
)
from stellargraph.layer import DeepGraphInfomax, GCN, link_classification, LinkEmbedding
from stellargraph import datasets
from code.node2vec_reg import Node2VecReg
from code.graphsage_reg import (
    GraphSAGEReg, 
    MeanAggregator,
    MaxPoolingAggregator,
    MeanPoolingAggregator,
    AttentionalAggregator
)
from code.gcn_reg import GCNReg

from sklearn.manifold import TSNE
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import accuracy_score, f1_score

_available_datasets = [
    'Cora',
    'CiteSeer',
    'PubMedDiabetes'
]

_optimization_algorithms = {
    'Adam' : keras.optimizers.Adam,
    'SGD' : keras.optimizers.SGD
}

_available_methods = [
    'node2vec',
    'GraphSAGE',
    'GCN',
    'DGI'
]

_graphSAGE_aggregators = {
    'mean': MeanAggregator,
    'mean_pool': MeanPoolingAggregator,
    'max_pool': MaxPoolingAggregator,
    'attention': AttentionalAggregator
}


def parse_arguments(parser = None):
    # Parses arguments
    if parser is None:
        parser = argparse.ArgumentParser()

    parser.add_argument('--dataset', type = str, default = 'Cora',
        choices = _available_datasets,
        help = "Dataset to perform training on. Available options: " + ','.join(_available_datasets)
    )
    parser.add_argument('--emb-size', type = int, default = 128,
        help = "Embedding dimension. Defaults to 128."
    )
    parser.add_argument('--reg-weight', type = float, default = 0.0,
        help = "Weight to use for L2 regularization. If norm_reg is True, then reg_weight/num_of_nodes is used instead."
    )
    parser.add_argument('--norm-reg', type = bool, default = False,
        help = "Boolean for whether to normalize the L2 regularization weight by the number of nodes in the graph. Defaults to false."
    )
    parser.add_argument('--method', type = str, default = 'node2vec',
        choices = _available_methods,
        help = "Algorithm to perform training on. Available options: " + ','.join(_available_methods)
    )
    parser.add_argument('--verbose', type = int, default = 1,
        help = 'Level of verbosity. Defaults to 1.'
    )

    # Optimization arguments:
    parser.add_argument('--epochs', type = int, default = 5,
        help = "Number of epochs through the dataset to be used for training."
    )
    parser.add_argument('--optimizer', type = str, default = 'Adam',
        choices = ['Adam', 'SGD'],
        help = "Optimization algorithm to use for training."
    )
    parser.add_argument('--learning-rate', type = float, default = 1e-3,
        help = "Learning rate to use for optimization."
    )
    parser.add_argument('--batch-size', type = int, default = 64,
        help = "Batch size used for training."
    )

    # Evaluation arguements:
    parser.add_argument('--train-split', type = float,
        nargs="*", 
        default = [0.01, 0.025, 0.05],
        help = "Percentage(s) to use for the training split when using the learned embeddings for downstream classification tasks."
    )
    parser.add_argument('--train-split-num', type = int, default = 25,
        help = "Decides the number of random training/test splits to use for evaluating performance. Defaults to 50."
    )
    parser.add_argument('--output-fname', type = str, default = None,
        help = "If not None, saves the hyperparameters and testing results to a .json file with filename given by the argument."
    )

    # Node2vec arguments, also used for GraphSAGE too
    parser.add_argument('--node2vec-p', type = float, default = 1.0,
        help = "Hyperparameter governing probability of returning to source node."
    )
    parser.add_argument('--node2vec-q', type = float, default = 1.0,
        help = "Hyperparameter governing probability of moving to a node away from the source node."
    )
    parser.add_argument('--node2vec-walk-number', type = int, default = 50,
        help = "Number of walks used to generate a sample for node2vec."
    )
    parser.add_argument('--node2vec-walk-length', type = int, default = 5,
        help = "Walk length to use for node2vec."
    )

    # DeepGraphInfomax arguments
    parser.add_argument('--dgi-sampler', type = str, default = 'fullbatch',
        choices = ['fullbatch', 'minibatch'],
        help = 'Specifies either a fullbatch or a minibatch sampling scheme for DGI.'
    )
    parser.add_argument('--gcn-activation', 
        nargs = "*",
        type = str,
        default = ['relu'],
        help = 'Determines the activations of each layer within a GCN. Defaults to a single layer with relu activation.'
    )

    # GraphSAGE arguments
    parser.add_argument('--graphSAGE-aggregator', type = str,
        default = 'mean',
        choices = ['mean', 'mean_pool', 'max_pool', 'attention'],
        help = 'Specifies the aggreagtion rule used in GraphSAGE. Defaults to mean pooling.'
    )
    parser.add_argument('--graphSAGE-nbhd-sizes', type = int, nargs="*", 
        default = [10, 5],
        help = 'Specify multiple neighbourhood sizes for sampling in GraphSAGE. Defaults to [10, 5].'
    )

    # Debugging arguments
    parser.add_argument('--tensorboard', default = False, action = 'store_true',
        help = 'If toggles, saves Tensorboard logs for debugging purposes.'
    )
    parser.add_argument('--visualize-embeds', default = None, type = str,
        help = 'If specified with a directory, saves an image of a TSNE 2D projection of the learned embeddings at the specified directory.'
    )
    parser.add_argument('--save-spectrum', default = None, type = str, 
        help = "If specifies, saves the spectrum of the learned embeddings output by the algorithm."
    )
    return parser.parse_args()


def load_data(str):
    if str not in _available_datasets:
        raise NameError("expected one of the following datasets: " + ','.join(_available_datasets))

    if (str == 'Cora'):
        G, subjects = datasets.Cora().load(largest_connected_component_only=True)
    elif (str == "CiteSeer"):
        G, subjects = datasets.CiteSeer().load(largest_connected_component_only=True)
    elif (str == "PubMedDiabetes"):
        G, subjects = datasets.PubMedDiabetes().load()

    return G, subjects


def linear_regression_eval(X_train, X_test, y_train, y_test, multi_class='ovr'):
    if multi_class not in ['ovr', 'multinomial']:
        multi_class = 'ovr'

    clf = LogisticRegressionCV(
        Cs=10, cv=5, scoring="accuracy", verbose=False, 
        multi_class=multi_class, max_iter=1000
    ).fit(X_train, y_train)
    y_pred = clf.predict(X_test)

    accuracy = accuracy_score(y_test, y_pred),
    macro_f1 = f1_score(y_test, y_pred, average = 'macro')
    micro_f1 = f1_score(y_test, y_pred, average = 'micro')

    return accuracy, macro_f1, micro_f1


def eval_over_splits(X, y, train_split=0.05, num_split=20):
    accuracy = [0]*num_split
    macro_f1 = [0]*num_split
    micro_f1 = [0]*num_split

    for i in range(num_split):
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, 
            train_size=train_split, 
            test_size=None, 
            stratify=y
        )

        accuracy[i], macro_f1[i], micro_f1[i] = linear_regression_eval(
            X_train, X_test, y_train, y_test
        )

    accuracy_dict = {'mean': np.mean(accuracy), 'sd': np.std(accuracy)}
    macro_dict = {'mean': np.mean(macro_f1), 'sd': np.std(macro_f1)}
    micro_dict = {'mean': np.mean(micro_f1), 'sd': np.std(micro_f1)}

    return {'train_split': train_split, 'accuracy': accuracy_dict, 'macro_f1': macro_dict, 'micro_f1': micro_dict}


def _normalize_regularization(reg_weight, num_of_nodes):
    return reg_weight/num_of_nodes


def _create_optimizer(optimizer, lr):
    return _optimization_algorithms[optimizer](learning_rate = lr)


def create_sampler(G, args):
    if (args.method in ['GCN', 'node2vec', 'GraphSAGE']):
        walker = BiasedRandomWalk(G, 
            n = args.node2vec_walk_number, 
            length = args.node2vec_walk_length, 
            p = args.node2vec_p, 
            q = args.node2vec_q
        )
        unsupervised_samples = UnsupervisedSampler(G, 
            nodes = list(G.nodes()), 
            walker = walker
        )
    elif (args.method == 'DGI'):
        pass

    return unsupervised_samples


def create_model_object(G, args):
    if (args.method == 'node2vec'):
        generator = Node2VecLinkGenerator(G, args.batch_size)
        method_class = Node2VecReg(
            emb_size = args.emb_size, 
            generator = generator, 
            reg_weight = args.reg_weight
        )

    elif (args.method == 'GraphSAGE'):
        generator = GraphSAGELinkGenerator(G, args.batch_size, args.graphSAGE_nbhd_sizes)
        method_class = GraphSAGEReg(
            layer_sizes = [args.emb_size]*len(args.graphSAGE_nbhd_sizes),
            generator = generator,
            aggregator = _graphSAGE_aggregators[args.graphSAGE_aggregator],
            activity_regularizer = keras.regularizers.l2(args.reg_weight)
        )

    elif (args.method == 'GCN'):
        generator = FullBatchLinkGenerator(G, method='gcn')
        method_class = GCNReg(
            layer_sizes = [args.emb_size]*len(args.gcn_activation), 
            generator = generator,
            activations = args.gcn_activation, 
            reg_weight = args.reg_weight,
        )

    return generator, method_class


def get_embeddings(G, subjects, embedding_model, args, generator=None):
    if (args.method == 'node2vec'):
        node_gen = Node2VecNodeGenerator(G, args.batch_size).flow(subjects.index)
        node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=args.verbose)

    elif (args.method == 'GraphSAGE'):
        node_gen = GraphSAGENodeGenerator(G, args.batch_size, args.graphSAGE_nbhd_sizes).flow(subjects.index)
        node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=args.verbose)

    elif (args.method == 'GCN'):
        node_gen = FullBatchNodeGenerator(G).flow(subjects.index)
        node_embeddings = embedding_model.predict(node_gen, workers=4, verbose=args.verbose)
        node_embeddings = node_embeddings.squeeze(0)

    elif (args.method == 'DGI'):
        node_embeddings = embedding_model.predict(
            generator.flow(subjects.index),
            workers=4,
            verbose=args.verbose
        )

    return node_embeddings


def main():
    # Handle arguments, Load the dataset, handle any hyperparameter normalizing
    args = parse_arguments()
    G, subjects = load_data(args.dataset)

    if args.norm_reg:
        args.reg_weight = _normalize_regularization(args.reg_weight, G.number_of_nodes())

    # Create the sampler and model class used for training
    if (args.method == 'DGI'):
        if (args.dgi_sampler == 'fullbatch'):
            generator = FullBatchNodeGenerator(G, sparse=False)
            base_model =  GCNReg(
                layer_sizes = [args.emb_size]*len(args.gcn_activation), 
                generator = generator,
                activations = args.gcn_activation, 
                reg_weight = args.reg_weight,
            )
        elif (args.dgi_sampler == 'minibatch'):
            generator = GraphSAGENodeGenerator(G, 
                batch_size=args.batch_size, 
                num_samples=args.graphSAGE_nbhd_sizes
            )
            base_model = GraphSAGEReg(
                layer_sizes=[args.emb_size]*len(args.graphSAGE_nbhd_sizes), 
                generator=generator,
                activity_regularizer = keras.regularizers.L2(args.reg_weight)
            )
# )

        corrupted_generator = CorruptedGenerator(generator)
        model_class = DeepGraphInfomax(base_model, corrupted_generator)
    
    if (args.method in ['GCN', 'node2vec', 'GraphSAGE']):
        unsupervised_samples = create_sampler(G, args)
        generator, model_class = create_model_object(G, args)
        
    x_inp, x_out = model_class.in_out_tensors() 

    # Forming the loss function used for training
    if (args.method in ['node2vec', 'GraphSAGE']):
        prediction = link_classification(
            output_dim = 1, 
            output_act = "sigmoid", 
            edge_embedding_method = "dot"
        )(x_out)
    elif (args.method == 'GCN'):
        prediction = LinkEmbedding(
            activation = "sigmoid", 
            method = "ip"
        )(x_out)
        prediction = keras.layers.Reshape((-1, ))(prediction)
    else:
        prediction = x_out

    model = keras.Model(inputs=x_inp, outputs=prediction) 

    # tf.nn.sigmoid_cross_entropy_with_logits if args.method == 'DGI' else keras.losses.binary_crossentropy
    model.compile(
        optimizer = _create_optimizer(args.optimizer, args.learning_rate), 
        loss = keras.losses.BinaryCrossentropy(from_logits=True if args.method == 'DGI' else False), 
        metrics = [keras.metrics.binary_accuracy]
    )

    # Perform training
    if (args.method == 'DGI'): 
        es = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=20)
        tb = tf.keras.callbacks.TensorBoard(log_dir = args.output_fname + "/logs" if args.output_fname is not None else "logs/")

        if args.tensorboard:
            callbacks = [es, tb]
        else:
            callbacks = [es]

        history = model.fit(
            corrupted_generator.flow(G.nodes()),
            epochs=args.epochs,
            verbose=args.verbose,
            use_multiprocessing=False,
            workers=4,
            callbacks=callbacks
        )
    
    if (args.method in ['node2vec', 'GraphSAGE']):
        if args.tensorboard:
            callbacks = [tf.keras.callbacks.TensorBoard(log_dir = args.output_fname + "/logs" if args.output_fname is not None else "logs/")]
        else:
            callbacks = None

        history = model.fit(
            generator.flow(unsupervised_samples),
            epochs=args.epochs, 
            verbose=args.verbose,
            use_multiprocessing=False,
            workers=4,
            shuffle=True,
            callbacks=callbacks
        )

    if (args.method == 'GCN'):
        batches = unsupervised_samples.run(args.batch_size)
        for i in range(args.epochs):
            print(f'Epoch: {i}/{args.epochs}')
            for batch in tqdm(batches):
                samples = generator.flow(
                    batch[0], targets = batch[1], use_ilocs = True
                )[0]
                model.train_on_batch(x = samples[0], y = samples[1])

    # Generate embeddings
    if (args.method == 'DGI'):
        x_inp_src, x_out_src = base_model.in_out_tensors()
        # Squeze out batch dimension for full batch models
        if generator.num_batch_dims() == 2:
            x_out_src = tf.squeeze(x_out_src, axis=0)

    if (args.method == 'node2vec'):
        x_inp_src = x_inp[0]
        x_out_src = x_out[0]

    if (args.method == 'GraphSAGE'):
        x_inp_src = x_inp[0::2]
        x_out_src = x_out[0]

    if (args.method == 'GCN'):
        x_inp_src = x_inp
        x_out_src = x_out

    embedding_model = keras.Model(inputs=x_inp_src, outputs=x_out_src)
    node_embeddings = get_embeddings(
        G=G, 
        subjects=subjects, 
        embedding_model=embedding_model, 
        args=args,
        generator=generator if args.method == 'DGI' else None
    )

    # Evaluate performance on training/test splits
    X = node_embeddings
    y = np.array(subjects)

    if args.visualize_embeds is not None:
        node_embeddings_2d = TSNE(n_components=2).fit_transform(node_embeddings)
        label_map = {l: i for i, l in enumerate(np.unique(subjects))}
        node_colours = [label_map[target] for target in subjects]

        fig, ax = plt.subplots(figsize=(7, 7))
        ax.scatter(
            node_embeddings_2d[:, 0],
            node_embeddings_2d[:, 1],
            c=node_colours,
            cmap="jet",
            alpha=0.7,
        )
        ax.set(aspect="equal", xlabel="$X_1$", ylabel="$X_2$")
        plt.savefig(f"{args.visualize_embeds}.png", )

    if args.save_spectrum is not None:
        np.savetxt(f"{args.save_spectrum}.csv", 
            scipy.linalg.svdvals(X),
            delimiter=","
        )

    output_dict = {**vars(args)}

    for splits in args.train_split:
        output_dict[f'LinReg_{splits}'] = eval_over_splits(
            X=X, y=y,
            train_split=splits,
            num_split=args.train_split_num
        )
    
        if (args.method == 'node2vec'):
            output_dict[f'LinRegNF_{splits}'] = eval_over_splits(
                X=np.concatenate((X, G.node_features()), axis=1), 
                y=y, 
                train_split=splits,
                num_split=args.train_split_num
            )

    if args.output_fname is not None:
        print('Saving results to file...')
        with open(args.output_fname + '.json', "w") as outfile:
            json.dump(output_dict, outfile)
    else:
        print(output_dict)


if __name__ == '__main__':
    main()
