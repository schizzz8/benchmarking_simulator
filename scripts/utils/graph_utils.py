#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import mxnet as mx
import numpy as np


def make_list(data):
    if isinstance(data, list):
        new_list = data
    elif isinstance(data, np.ndarray):
        new_list = data.tolist()
    else:
        new_list = [data]

    return new_list


def network(input_variable_list,
            n_hidden,
            hidden_activations,
            network_id):
    input_variable_list = make_list(input_variable_list)
    n_hidden = make_list(n_hidden)
    hidden_activations = make_list(hidden_activations)

    assert len(n_hidden) == len(hidden_activations)
    assert len(n_hidden) > 0

    n_layers = len(n_hidden)
    layers = [None] * (n_layers + 1)

    nid = network_id

    # setup input variables
    layers[0] = mx.sym.Concat(*input_variable_list, name=nid + '_data_in')

    for i in range(n_layers):
        prev_layer = layers[i]
        new_layer = mx.sym.FullyConnected(prev_layer,
                                          num_hidden=n_hidden[i],
                                          name=nid + '_fullyconnected_' + str(i + 1))

        if hidden_activations[i] is not None:
            layers[i + 1] = mx.sym.Activation(data=new_layer,
                                              act_type=hidden_activations[i],
                                              name=nid + '_activation_' + str(i + 1))
        else:
            layers[i + 1] = new_layer

    layers = [l for l in layers if l is not None]
    output = layers[-1]

    return output


def convnet(n_outputs,
            n_hidden,
            hidden_activations,
            network_id,
            output_activation='sigmoid',
            n_convolution_filters=[],
            convolution_activations=[],
            convolution_kernels=[],
            pool_types=[],
            pool_kernels=[],
            convolution_obs_names=[]):
    n_hidden = make_list(n_hidden)
    hidden_activations = make_list(hidden_activations)

    assert len(n_hidden) == len(hidden_activations)

    n_convolution_filters = make_list(n_convolution_filters)
    convolution_activations = make_list(convolution_activations)
    convolution_kernels = make_list(convolution_kernels)
    pool_types = make_list(pool_types)
    pool_kernels = make_list(pool_kernels)
    convolution_obs_names = make_list(convolution_obs_names)

    # check size list consistency
    assert len(convolution_obs_names) > 0
    assert len(n_convolution_filters) == len(convolution_activations)
    assert len(n_convolution_filters) == len(convolution_kernels)
    assert len(n_convolution_filters) == len(pool_types)
    assert len(n_convolution_filters) == len(pool_kernels)
    n_convolution_networks = len(convolution_obs_names)
    n_convolutions_per_data = len(n_convolution_filters)

    convolution_layers = [None] * n_convolution_networks

    for i in range(n_convolution_networks):
        # setup input variable for each separate convolution network
        convolution_data = mx.sym.Variable(convolution_obs_names[i])
        convolution_layers[i] = [None] * (n_convolutions_per_data + 1)

        convolution_layers[i][0] = convolution_data

        for j in range(n_convolutions_per_data):
            data = convolution_layers[i][j]
            convolution = mx.sym.Convolution(data=data, kernel=convolution_kernels[j],
                                             num_filter=n_convolution_filters[j],
                                             name=network_id + '_convolution_' + str(i + 1) + '_conv_' + str(j + 1))
            activation = mx.sym.Activation(data=convolution, act_type=convolution_activations[j],
                                           name=network_id + '_activation_' + str(i + 1) + '_conv_' + str(j + 1))

            if pool_types[j] is not None:
                pooling = mx.sym.Pooling(data=activation, pool_type=pool_types[j],  kernel=pool_kernels[j],
                                         stride=pool_kernels[j], name=network_id + '_pooling_' + str(i + 1) +
                                         '_conv_' + str(j + 1))

                convolution_layers[i][j + 1] = pooling
            else:
                convolution_layers[i][j + 1] = activation

        convolution_layers[i][n_convolutions_per_data] = mx.sym.Flatten(
            data=convolution_layers[i][n_convolutions_per_data], name=network_id + '_flatten_' + str(i + 1))

    convolution_layers = np.array([l for l in convolution_layers if l is not None])
    convoluted_data = list(convolution_layers[:, -1])

    hiddens = [*n_hidden, n_outputs]
    activations = [*hidden_activations, output_activation]

    output = network(convoluted_data, hiddens, activations, network_id)

    return output
