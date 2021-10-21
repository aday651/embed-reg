import numpy as np
import argparse
import json
from stellargraph.mapper.corrupted import CorruptedGenerator
import tensorflow as tf
from tensorflow import keras 
from tqdm import tqdm

from stellargraph.data import BiasedRandomWalk, UnsupervisedSampler, EdgeSplitter
from stellargraph.mapper import (
    Node2VecLinkGenerator, 
    Node2VecNodeGenerator, 
    GraphSAGENodeGenerator, 
    GraphSAGELinkGenerator,
    FullBatchLinkGenerator,
    CorruptedGenerator,
    FullBatchNodeGenerator
)
from stellargraph.layer import DeepGraphInfomax, link_classification, LinkEmbedding
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

from sklearn.pipeline import Pipeline 
from sklearn.preprocessing import StandardScaler
from sklearn.linear_model import LogisticRegressionCV
from sklearn.metrics import roc_auc_score, average_precision_score

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
    'GCN',
    'GraphSAGE',
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
    parser.add_argument('--test-split', type = float, default = 0.1,
        help = "Split of edge/non-edge set to be used for testing."
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
    parser.add_argument('--gcn-activation', 
        nargs = "*",
        type = str,
        default = ['relu'],
        help = 'Specifies layers in terms of their output activation (either relu or linear), with the number of arguments determining the length of the GCN. Defaults to a single layer with relu activation.'
    )

    # GraphSAGE arguments
    parser.add_argument('--graphSAGE-aggregator', type = str,
        default = 'mean',
        choices = ['mean', 'mean_pool', 'max_pool', 'attention'],
        help = 'Specifies the aggreagtion rule used in GraphSAGE. Defaults to mean pooling.'
    )
    parser.add_argument('--graphSAGE-nbhd-sizes', type = int, nargs="*", 
        default = [10, 5],
        help = 'Specify multiple neighbourhood sizes for sampling in GraphSAGE. Defaults to [25, 10].'
    )

    # Debugging arguments
    parser.add_argument('--tensorboard', default = False, action = 'store_true',
        help = 'If toggles, saves Tensorboard logs for debugging purposes.'
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


def _normalize_regularization(reg_weight, num_of_nodes):
    return reg_weight/num_of_nodes


def _create_optimizer(optimizer, lr):
    return _optimization_algorithms[optimizer](learning_rate = lr)


def create_sampler(G, args):
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

    return unsupervised_samples


def _create_generator(args):
    if (args.method == 'node2vec'):
        def _generator(G):
            return Node2VecLinkGenerator(G, args.batch_size)
    elif (args.method == 'GraphSAGE'):
        def _generator(G):
            return GraphSAGELinkGenerator(
                G, args.batch_size, args.graphSAGE_nbhd_sizes
            )
    elif (args.method == 'GCN'):
        def _generator(G):
            return FullBatchLinkGenerator(G, method = "gcn")
    elif (args.method == 'DGI'):
        def _generator(G):
            return FullBatchNodeGenerator(G, sparse = False)

    return _generator


def _create_model(generator, args):
    if (args.method == 'node2vec'):
        model_fn = Node2VecReg(
            emb_size = args.emb_size, 
            generator = generator, 
            reg_weight = args.reg_weight
        )
    elif (args.method == 'GraphSAGE'):
        model_fn = GraphSAGEReg(
            layer_sizes = [args.emb_size]*len(args.graphSAGE_nbhd_sizes),
            generator = generator,
            aggregator = _graphSAGE_aggregators[args.graphSAGE_aggregator],
            activity_regularizer = keras.regularizers.l2(args.reg_weight)
        )
    elif (args.method == 'GCN'):
        model_fn = GCNReg(
            layer_sizes = [args.emb_size]*len(args.gcn_activation), 
            generator = generator,
            activations = args.gcn_activation, 
            reg_weight = args.reg_weight,
        )

    return model_fn


def _create_dgi_model(generator, corrupted_generator, args):
    base_model = GCNReg(
        layer_sizes = [args.emb_size]*len(args.gcn_activation), 
        generator = generator,
        activations = args.gcn_activation, 
        reg_weight = args.reg_weight,
    )

    model_fn = DeepGraphInfomax(base_model, corrupted_generator)

    return base_model, model_fn


def _create_inputs_outputs(model_fn, args):
    x_inp, x_out = model_fn.in_out_tensors()

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
    elif (args.method == 'DGI'):
        prediction = x_out

    return x_inp, x_out, prediction


def _embedding_fn(G_train, train_gen, x_inp, x_out, args, attach_node_features = False):
    if (args.method == 'node2vec'):
        x_inp_src = x_inp[0] 
        x_out_src = x_out[0] 
        embedding_model = keras.Model(
            inputs = x_inp_src, 
            outputs = x_out_src
        )

        node_gen = Node2VecNodeGenerator(
            G_train, args.batch_size
        ).flow(list(G_train.nodes())) 
        
        node_embeddings = embedding_model.predict(
            node_gen, workers = 1, verbose = 0
        )
    elif (args.method == 'GraphSAGE'):
        x_inp_src = x_inp[0::2]
        x_out_src = x_out[0]
        embedding_model = keras.Model(
            inputs = x_inp_src, 
            outputs = x_out_src
        )

        node_gen = GraphSAGENodeGenerator(
            G_train, args.batch_size, args.graphSAGE_nbhd_sizes
        ).flow(list(G_train.nodes()))

        node_embeddings = embedding_model.predict(
            node_gen, workers = 1, verbose = 0
        )
    elif (args.method == 'GCN'):
        embedding_model = keras.Model(inputs=x_inp, outputs=x_out)
        node_embeddings = embedding_model.predict(
            train_gen.flow(list(zip(
                list(G_train.nodes()), list(G_train.nodes())
            )))
        )
        node_embeddings = node_embeddings[0][:, 0, :]
    else:
        embedding_model = keras.Model(inputs=x_inp, outputs=x_out)
        node_embeddings = embedding_model.predict(
            train_gen.flow(list(G_train.nodes())),
            workers = 4, verbose = args.verbose
        )

    if not attach_node_features:
        def get_embedding(u):
            u_index = list(G_train.nodes()).index(u)
            return node_embeddings[u_index]
    else:
        def get_embedding(u):
            u_index = list(G_train.nodes()).index(u)
            return np.concatenate((
                node_embeddings[u_index],
                G_train.node_features()[u_index]
            ))

    return get_embedding


def hadamard(u, v):
    return u * v


def l1_op(u, v):
    return np.abs(u - v)


def l2_op(u, v):
    return (u - v)**2


binary_ops = {
    'hadamard': hadamard, 
    'l1_op': l1_op, 
    'l2_op': l2_op
}


def main():
    # Handle arguments, Load the dataset, handle any hyperparameter normalizing
    args = parse_arguments()
    print(args)
    G, _ = load_data(args.dataset)

    if args.norm_reg:
        args.reg_weight = _normalize_regularization(
            args.reg_weight, 
            G.number_of_nodes()
        )

    # Create the training and test graphs
    edge_split_test = EdgeSplitter(G)
    G_test, examples_test, labels_test = edge_split_test.train_test_split(
        p = args.test_split, 
        method = 'global'
    )

    edge_split_train = EdgeSplitter(G_test)
    G_train, examples_train, labels_train = edge_split_train.train_test_split(
        p = args.test_split, 
        method = 'global'
    )

    if (args.method != 'DGI'):
        # Create sampler for the training samples
        unsupervised_samples = create_sampler(G_train, args)
        # Create link generator for the training and test graphs
        _generator = _create_generator(args)
        train_gen = _generator(G_train)
        test_gen = _generator(G_test)
    else:
        # Define the generator for the training and test graphs, along
        # with the corrupted generator for the DGI training
        _generator = _create_generator(args)
        train_gen = _generator(G_train)
        test_gen = _generator(G_test)
        corrupted_generator = CorruptedGenerator(train_gen)

    # Define the model class on the training set
    if (args.method != 'DGI'):
        model_fn = _create_model(train_gen, args)
    else:
        base_model, model_fn = _create_dgi_model(train_gen, corrupted_generator, args)

    x_inp, x_out, prediction = _create_inputs_outputs(model_fn, args)

    # Train on the training set
    model = keras.Model(inputs = x_inp, outputs = prediction)
    model.compile(
        optimizer = _create_optimizer(args.optimizer, args.learning_rate), 
        loss = keras.losses.BinaryCrossentropy(from_logits=True if args.method == 'DGI' else False),
        metrics = [keras.metrics.binary_accuracy]
    )

    if (args.method in ['node2vec', 'GraphSAGE']):
        model.fit(
            train_gen.flow(unsupervised_samples),
            epochs = args.epochs,
            verbose = args.verbose,
            use_multiprocessing = False, 
            workers = 4,
            shuffle = True
        )

    elif (args.method == 'GCN'):
        batches = unsupervised_samples.run(args.batch_size)
        for i in range(args.epochs):
            print(f'Epoch: {i}/{args.epochs}')
            for batch in tqdm(batches):
                samples = train_gen.flow(
                    batch[0], targets = batch[1], use_ilocs = True
                )[0]
                model.train_on_batch(x = samples[0], y = samples[1])

    elif (args.method == 'DGI'):
        es = keras.callbacks.EarlyStopping(monitor="loss", min_delta=0, patience=20)
        history = model.fit(
            corrupted_generator.flow(G.nodes()),
            epochs = args.epochs,
            verbose = args.verbose,
            use_multiprocessing = False,
            workers = 4,
            callbacks= [es]
        )

    # Evaluate using model.evaluate(test_graph)
    output_dict = {**vars(args)}

    # figure out how this is foratted... model.evaluate(test_flow)
    if (args.method != 'DGI'):
        output_dict['inner_prod'] = dict(zip(
            model.metrics_names,
            model.evaluate(test_gen.flow(examples_test, labels_test))
        ))

    # Evaluate using higher dimensional link embeddings 
    if (args.method != 'DGI'):
        embedding_fn = _embedding_fn(
            G_train = G_train,
            train_gen = train_gen, 
            x_inp = x_inp, 
            x_out = x_out, 
            args = args
        )
    else:
        x_inp_src, x_out_src = base_model.in_out_tensors()
        # Squeze out batch dimension for full batch models
        if train_gen.num_batch_dims() == 2:
            x_out_src = tf.squeeze(x_out_src, axis=0)

        embedding_fn = _embedding_fn(
            G_train = G_train, 
            train_gen = train_gen,
            x_inp = x_inp_src, 
            x_out = x_out_src, 
            args = args
        )
    
    for op_name, binary_op in binary_ops.items():
        features_train = [
            binary_op(embedding_fn(src), embedding_fn(dst)) 
            for src, dst in examples_train
        ]
        features_test = [
            binary_op(embedding_fn(src), embedding_fn(dst)) 
            for src, dst in examples_test
        ]

        clf = Pipeline(steps=[
            ("sc", StandardScaler()), 
            ("clf", LogisticRegressionCV(
            Cs=10, cv=5, scoring="roc_auc", verbose=False, max_iter=1000))
        ])

        clf.fit(features_train, labels_train)
        preds = clf.predict_proba(features_test)

        pos_col = list(clf.classes_).index(1)
        output_dict[op_name] = {
            'roc_auc': roc_auc_score(labels_test, preds[:, pos_col]),
            'pr_auc': average_precision_score(labels_test, preds[:, pos_col])
        }

    if (args.method == 'node2vec'):
        embedding_fn_nf = _embedding_fn(
            G_train = G_train,
            train_gen = train_gen, 
            x_inp = x_inp, 
            x_out = x_out, 
            args = args,
            attach_node_features = True
        )

        for op_name, binary_op in binary_ops.items():
            features_train = [
                binary_op(embedding_fn_nf(src), embedding_fn_nf(dst)) 
                for src, dst in examples_train
            ]
            features_test = [
                binary_op(embedding_fn_nf(src), embedding_fn_nf(dst)) 
                for src, dst in examples_test
            ]

            clf = Pipeline(steps=[
                ("sc", StandardScaler()), 
                ("clf", LogisticRegressionCV(
                Cs=10, cv=5, scoring="roc_auc", verbose=False, max_iter=1000))
            ])

            clf.fit(features_train, labels_train)
            preds = clf.predict_proba(features_test)

            pos_col = list(clf.classes_).index(1)
            output_dict[op_name + '_NF'] = {
                'roc_auc': roc_auc_score(labels_test, preds[:, pos_col]),
                'pr_auc': average_precision_score(labels_test, preds[:, pos_col])
            }


    if args.output_fname is not None:
        print('Saving results to file...')
        with open(args.output_fname + '.json', "w") as outfile:
            json.dump(output_dict, outfile)
    else:
        print(output_dict)


if __name__ == '__main__':
    main()
