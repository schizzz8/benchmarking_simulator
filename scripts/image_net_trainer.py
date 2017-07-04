#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import argparse
import os
import tarfile
import warnings

from IPython import embed
from random import shuffle
from shutil import rmtree
from six.moves import urllib
from scipy.misc import imread, imresize

from random import shuffle
import mxnet as mx
import numpy as np
import pickle as pickle
import xml.etree.ElementTree as xml

from utils.graph_utils import convnet

base_url = 'http://image-net.org/'
synset_word_map_url = base_url + 'archive/words.txt'
synset_list_url = base_url + 'api/text/imagenet.bbox.obtain_synset_list'
synset_image_name_url_map = base_url + \
    'api/text/imagenet.synset.geturls.getmapping?wnid={}'
synset_image_bbox = base_url + '/downloads/bbox/bbox/{}.tar.gz'

script_path = os.path.dirname(os.path.realpath(__file__))

desired_classes = ['chair', 'table', 'fridge', 'cat', 'dog', 'wall',
                   'fire extinguisher', 'furniture', 'closet', 'door',
                   'office', 'person']


def mkdir(name):
    if not os.path.exists(name):
        os.mkdir(name)

    return os.path.abspath(name)


def list_files(path, ext=None):
    files = []

    for p, sd, fs in os.walk(path):
        for f in fs:
            if ext is None or f.endswith('.' + ext):
                files.append(os.path.join(p, f))

    return files


def retrieve_meta():
    print('Downloading meta-data')
    synset_list_file, _ = urllib.request.urlretrieve(synset_list_url)
    synset_list = [s.strip() for s in open(synset_list_file).readlines()]

    word_map_file, _ = urllib.request.urlretrieve(synset_word_map_url)
    synset_dict = {k.strip(): v.strip().lower().split(',') for k, v in (
        s.strip().split('\t') for s in open(word_map_file).readlines())
        if k in synset_list}
    filtered_synset_dict = dict()

    print('Meta-data downloaded. Filtering.')
    for k, v in synset_dict.items():
        for c in desired_classes:
            if c in v:
                filtered_synset_dict.update({k: c})
                break

    print('Meta-data filtered. Available: {} synsets'.format(
        len(filtered_synset_dict)))

    return filtered_synset_dict


def one_hot_labels(class_label):
    one_hot = np.array(
        [1. if l == class_label else 0. for l in desired_classes])
    return one_hot


def generate_dataset(meta, dataset_path, batch_size, test_percentage=0.2):
    print('Generating dataset')

    try:
        print('Deleting old dataset directory, if needed.')
        rmtree(dataset_path, ignore_errors=False, onerror=None)
    except Exception:
        pass

    dataset_path = mkdir(dataset_path)
    class_path = {c: mkdir(os.path.join(dataset_path, c))
                  for c in desired_classes}
    image_path = {c: mkdir(os.path.join(
        class_path[c], 'images')) for c in desired_classes}
    bbox_path = {c: mkdir(os.path.join(
        class_path[c], 'bboxes')) for c in desired_classes}

    print('Retrieving image urls')
    synset_image_name_map = dict()
    training_dataset = dict()
    testing_dataset = dict()

    for k, v in meta.items():
        synset_image_name_url_map_file, _ = urllib.request.urlretrieve(
            synset_image_name_url_map.format(k))
        id_image_url_map = {k.strip(): v.strip() for k, v in (s.strip().split(
            ' ') for s in open(synset_image_name_url_map_file).readlines() if len(s.strip()) > 0)}
        synset_image_name_map.update(id_image_url_map)
        print('\t{}:\t\t{}\tnum. URLs {}'.format(
            v, k, len(id_image_url_map.keys())))

        bbox_id_path = os.path.join(
            class_path[v], 'bboxes_{}.tar.gz'.format(k))
        print('\tDownloading Bounding Box')
        synset_image_bbox_file, _ = urllib.request.urlretrieve(
            synset_image_bbox.format(k), bbox_id_path)
        tar = tarfile.open(bbox_id_path)
        tar.extractall(path=bbox_path[v])
        tar.close()
        print('\tDone')

    print('Pickle-dumpling meta, desired_classes to training_set.txt')
    with open(os.path.join(dataset_path, 'training_set.txt'), 'wb') as f:
        pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
        pickle.dump(desired_classes, f, protocol=pickle.HIGHEST_PROTOCOL)

    training_batch_counter = 0
    testing_batch_counter = 0
    all_files = []

    for c in desired_classes:
        print('Preparing class: {}'.format(c))
        all_files += [(f, c) for f in list_files(bbox_path[c], ext='xml')]

    shuffle(all_files)

    test_start_index = round(test_percentage * len(all_files))
    testing_files = [d[0] for d in all_files[-test_start_index:]]
    num_testing = len(testing_files)

    print('Dataset split in {} files for training, {} for testing'.format(
        len(all_files) - num_testing, num_testing))

    shuffle(all_files)

    for f_data in all_files:
        f = f_data[0]
        c = f_data[1]
        xf = xml.parse(f)
        image_id = xf.find('filename').text

        if image_id not in synset_image_name_map.keys():
            continue

        image_ext = os.path.splitext(synset_image_name_map[image_id])[1]
        image_id_path = os.path.join(image_path[c], image_id + image_ext)

        try:
            image_file, _ = urllib.request.urlretrieve(
                synset_image_name_map[image_id], image_id_path)
        except Exception:
            continue

        print('Downloading', image_id_path)

        size = xf.find('size')
        width = float(size.find('width').text)
        height = float(size.find('height').text)
        objs = xf.findall('object')

        for i, o in enumerate(objs):
            bndbox = o.find('bndbox')
            x_min = float(bndbox.find('xmin').text) / width
            x_max = float(bndbox.find('xmax').text) / width
            y_min = float(bndbox.find('ymin').text) / height
            y_max = float(bndbox.find('ymax').text) / height
            bbox = (x_min, x_max, y_min, y_max)
            image_descriptor = dict(path=image_id_path, bbox=bbox, label=one_hot_labels(c))

            if f in testing_files:
                testing_dataset.update({image_id + '_{}'.format(i): image_descriptor})
                testing_batch_counter += 1
            else:
                training_dataset.update({image_id + '_{}'.format(i): image_descriptor})
                training_batch_counter += 1

            if training_batch_counter == batch_size or testing_batch_counter == batch_size:
                if training_batch_counter == batch_size:
                    saving_filename = 'training_set.txt'
                    dump_dataset = training_dataset
                else:
                    saving_filename = 'testing_set.txt'
                    dump_dataset = testing_dataset

                print('Pickle-dumpling dataset batch to {}'.format(saving_filename))

                with open(os.path.join(dataset_path, saving_filename), 'a+b') as f:
                    pickle.dump(dump_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)

                if training_batch_counter == batch_size:
                    training_dataset.clear()
                    training_batch_counter = 0
                else:
                    testing_dataset.clear()
                    testing_batch_counter = 0


def generate_model(batch_size, image_width, image_height, context=mx.cpu(), render=False):
    # 4 for bounding_boxes
    num_outputs = len(desired_classes) + 4

    graph = convnet(num_outputs,
                    [4 * num_outputs, 4 * num_outputs],
                    ['relu', 'relu'],
                    'sem_labeling',
                    output_activation=None,
                    n_convolution_filters=[30, 60, 70, 70, 70],
                    convolution_activations=['relu', 'relu', 'relu', 'relu', 'relu'],
                    convolution_kernels=[(6, 6), (5, 5), (4, 4), (3, 3), (2, 2)],
                    pool_types=['max', 'max', 'max', 'max', 'max'],
                    pool_kernels=[(3, 3), (3, 3), (2, 2), (2, 2), (2, 2)],
                    convolution_obs_names=['rgb_image'])

    classes = mx.sym.slice_axis(graph, axis=1, begin=0, end=len(desired_classes))
    classes = mx.sym.SoftmaxOutput(data=graph, label=mx.sym.Variable('class_label'),
                                   multi_output=True, use_ignore=True, ignore_label=len(desired_classes),
                                   name='sem_labeling_class_out')

    bounding_boxes = mx.sym.slice_axis(graph, axis=1, begin=len(desired_classes), end=num_outputs)
    bounding_boxes = mx.sym.Activation(data=bounding_boxes, act_type='sigmoid')
    bounding_boxes = mx.sym.MAERegressionOutput(data=bounding_boxes, label=mx.sym.Variable('bbox_label'),
                                                name='sem_labeling_bbox_out')

    output = mx.sym.Group([classes, bounding_boxes])

    if render:
        gviz = mx.viz.plot_network(output)
        gviz.render(filename='graph', cleanup=True)

    image_shape = (batch_size, 3, image_width, image_height)
    input_shapes = {'rgb_image': image_shape, 'class_label': (batch_size,),
                    'bbox_label': (batch_size, 4)}
    executor = output.simple_bind(ctx=context, grad_req='write', **input_shapes)

    param_initializer = mx.initializer.MSRAPrelu()
    data_args = ['rgb_image', 'class_label', 'bbox_label']
    param_names = executor.arg_dict.keys() - set(data_args)
    key_index = {key: idx for idx, key in enumerate(output.list_arguments())}

    attrs = output.attr_dict()
    for key, array in executor.arg_dict.items():
        if key not in data_args:
            desc = mx.initializer.InitDesc(key, attrs.get(key, None))
            param_initializer(desc, array)

    return executor, output, data_args, param_names, key_index


def fix_dataset_batches(meta, desired_classes, dataset, batch_size, dataset_path):
    dataset_keys = dataset.keys()
    print(dataset_keys)
    dataset_size = len(dataset_keys)

    if dataset_size < batch_size:
        print('Impossible to fix dataset, not enough data for batch size')
        return False
    else:
        batch_counter = 0

        with open(os.path.join(dataset_path, 'training_set.txt'), 'wb') as f:
            pickle.dump(meta, f, protocol=pickle.HIGHEST_PROTOCOL)
            pickle.dump(desired_classes, f, protocol=pickle.HIGHEST_PROTOCOL)

            new_dataset = dict()
            for i, k in enumerate(dataset_keys):
                new_dataset[k] = dataset[k]

                if i + 1 % batch_size == 0:
                    pickle.dump(new_dataset, f, protocol=pickle.HIGHEST_PROTOCOL)
                    new_dataset.clear()
                    batch_counter += 1

        print('Dataset separated in {} batches'.format(batch_counter))
        return True


def prepare_dataset_batch(dataset_batch, image_width, image_height):
    values = dataset_batch.values()
    batch_size = len(values)

    batch = {'rgb_image': np.zeros((batch_size, 3, image_width, image_height)),
             'class_label': np.zeros(batch_size),
             'bbox_label': np.zeros((batch_size, 4))}

    for i, v in enumerate(values):
        try:
            image = imread(v['path'], flatten=False, mode='RGB')
            image = imresize(image, (image_width, image_height))
            image = np.array(image, dtype=np.float32)
            mean = np.array([123.68, 116.779, 103.939]).reshape(3, 1, 1)  # (R, G, B)

            batch['rgb_image'][i, :, :, :] = image.reshape(3, image_width, image_height) - mean
            batch['class_label'][i] = np.where(v['label'] >= 1.)[0]
            batch['bbox_label'][i, :] = np.array(v['bbox'])
        except OSError:
            batch['rgb_image'][i, :, :, :] = np.zeros((3, image_width, image_height))
            batch['class_label'][i] = len(desired_classes)
            batch['bbox_label'][i, :] = np.zeros(4)

    return batch


def test_error(model, testing_data, batch_size=50):
    testing_inputs = {'rgb_image': testing_data['rgb_image']}
    testing_ground_truth = {'class_probabilities': testing_data['class_label'],
                            'bounding_boxes': testing_data['bbox_label']}

    test_set_size = testing_inputs['rgb_image'].shape[0]
    sq_errors = {'class_probabilities': np.zeros(test_set_size), 'bounding_boxes': np.zeros(test_set_size)}
    inv_n_classes = 1 / len(desired_classes)

    for i in range(0, test_set_size, batch_size):
        end = min(i + batch_size, test_set_size)
        batch = {'rgb_image': testing_inputs['rgb_image'][i:end]}

        predictions = predict(model, batch)

        for k in predictions.keys():
            label = testing_ground_truth[k][i:end]

            if k == 'class_probabilities':
                pred = np.argmax(predictions[k])
                sq_errors[k][i:end] += np.abs((label - pred) * inv_n_classes)
            else:
                pred = predictions[k]
                sq_errors[k][i:end] += np.average(np.abs(label - pred), axis=1)

    avg_errors = {'class_probabilities': np.average(sq_errors['class_probabilities']),
                  'bounding_boxes': np.average(sq_errors['bounding_boxes'])}

    return avg_errors


def train_model(model, graph, data_args, param_names, key_index, dataset_path, save_model, batch_size,
                image_width, image_height, iterations=1, learning_rate=0.01, optimizer='sgd'):
    load = True
    optimizer = mx.optimizer.create(optimizer, learning_rate=learning_rate, rescale_grad=1. / batch_size, wd=0.0001)
    updater = mx.optimizer.get_updater(optimizer)

    testing_set = dict()
    with open(os.path.join(dataset_path, 'testing_set.txt'), 'rb') as f:
        print('Loading testing set')

        while True:
            try:
                testing_set.update(pickle.load(f))
            except EOFError:
                print('Finished reading testing dataset')
                break

    testing_data = prepare_dataset_batch(testing_set, image_width, image_height)

    while load:
        print('Loading training set')

        with open(os.path.join(dataset_path, 'training_set.txt'), 'rb') as f:
            load = False
            meta = pickle.load(f)
            desired_classes = pickle.load(f)

            while True:
                try:
                    dataset = pickle.load(f)
                except IOError:
                    print('Finished reading training dataset')
                    break

                if len(dataset.keys()) != batch_size:
                    warnings.warn("[WARNING] dataset batch size mismatch; attempting to fix")

                    fixed = fix_dataset_batches(meta, desired_classes, batch_size, dataset_path)

                    load = fixed
                    break

                print('Batch update')
                train_inputs = prepare_dataset_batch(dataset, image_width, image_height)

                for i in range(iterations):
                    print('Iteration {}'.format(i + 1))
                    model.forward(is_train=True, **train_inputs)
                    model.backward()

                    for key in param_names:
                        updater(key_index[key], model.grad_dict[key], model.arg_dict[key])

                avg_errors = test_error(model, train_inputs, batch_size=batch_size)
                print('Training avg. errors:\n\tclasses: {}\n\tbounding boxes: {}'.format(
                    avg_errors['class_probabilities'], avg_errors['bounding_boxes']))

                avg_errors = test_error(model, testing_data, batch_size=batch_size)
                print('Testing avg. errors:\n\tclasses: {}\n\tbounding boxes: {}'.format(
                    avg_errors['class_probabilities'], avg_errors['bounding_boxes']))

                if save_model is not None:
                    m_path = os.path.join(save_model, 'imagenet.model')
                    print('Saving model {}'.format(m_path))

                    with open(m_path, 'wb') as sf:
                        pickle.dump(model.arg_dict, sf, protocol=pickle.HIGHEST_PROTOCOL)
                        pickle.dump(graph, sf, protocol=pickle.HIGHEST_PROTOCOL)
                        pickle.dump(data_args, sf, protocol=pickle.HIGHEST_PROTOCOL)
                        pickle.dump(param_names, sf, protocol=pickle.HIGHEST_PROTOCOL)
                        pickle.dump(key_index, sf,  protocol=pickle.HIGHEST_PROTOCOL)
                        pickle.dump(batch_size, sf, protocol=pickle.HIGHEST_PROTOCOL)
                        pickle.dump(image_width, sf, protocol=pickle.HIGHEST_PROTOCOL)
                        pickle.dump(image_height, sf, protocol=pickle.HIGHEST_PROTOCOL)
                        pickle.dump(desired_classes, sf, protocol=pickle.HIGHEST_PROTOCOL)

    return model


def load_model(path, context=mx.cpu()):
    m_path = os.path.join(path, 'imagenet.model')
    print('Loading model {}'.format(m_path))

    with open(m_path, 'rb') as f:
        arg_dict = pickle.load(f)
        graph = pickle.load(f)
        data_args = pickle.load(f)
        param_names = pickle.load(f)
        key_index = pickle.load(f)
        batch_size = pickle.load(f)
        image_width = pickle.load(f)
        image_height = pickle.load(f)
        desired_classes = pickle.load(f)

    image_shape = (batch_size, 3, image_width, image_height)
    input_shapes = {'rgb_image': image_shape, 'class_label': (batch_size, len(desired_classes)),
                    'bbox_label': (batch_size, 4)}
    executor = graph.simple_bind(ctx=context, grad_req='write', **input_shapes)

    for k, v in arg_dict.items():
        if k not in data_args:
            v.copyto(executor.arg_dict[k])

    return executor, graph, data_args, param_names, key_index


def predict(model, inputs):
    predict_shapes = {'rgb_image': inputs['rgb_image'].shape}
    predict_model = model.reshape(partial_shaping=True, allow_up_sizing=False, **predict_shapes)

    predictions = predict_model.forward(is_train=False, **inputs)

    return {'class_probabilities': predictions[0].asnumpy(), 'bounding_boxes': predictions[1].asnumpy()}


if __name__ == '__main__':
    p = argparse.ArgumentParser(
        description='Helper for downloading and training from ImageNet.')
    p.add_argument('--dataset', '-d', action='store_true',
                   help='Download and generate dataset (default: false).')
    p.add_argument(
        '--dataset-path', '-dp',
        help='Path for storing the dataset. If not set, will use {}'
        .format(os.path.join(script_path, 'dataset')))
    p.add_argument('--test-percentage', '-tp', type=float, default=0.2,
                   help='Percentage of dataset used for testing (useful only at dataset creation time)')
    p.add_argument('--load-model', '-lm', help='Path for the model to load')
    p.add_argument('--save-model', '-sm', help='Path for saving the model')
    p.add_argument('--train-model', '-tm', action='store_true',
                   help='Train model (default: false).')
    p.add_argument('--image-width', '-iw', type=int, default=200,
                   help='Image resize-width for training (default: 200)')
    p.add_argument('--image-height', '-ih', type=int, default=200,
                   help='Image resize-height for training (default: 200)')
    p.add_argument('--batch-size', '-bs', type=int, default=50,
                   help='Training batch size (default: 50)')
    p.add_argument('--learning-iterations', '-li', type=int, default=10,
                   help='Training iterations (default: 10)')
    p.add_argument('--learning-rate', '-lr', type=float, default=0.001,
                   help='Learning rate (default: 0.001)')
    p.add_argument('--optimizer', '-op', type=str, default='sgd',
                   help='Optimizer (e.g., sgd, adam, etc., default: sgd)')
    p.add_argument('--context', '-ctx', type=str, default='cpu',
                   help='Context (cpu/gpu, default: cpu)')
    p.add_argument('--render-graph', '-rg', action='store_true', default=False,
                   help='Draw graph.pdf showing the graph (default: False)')

    args = p.parse_args()
    dataset_path = os.path.join(script_path, 'dataset')
    context = mx.cpu()

    if args.context == 'gpu':
        context = mx.gpu()

    if args.dataset_path is not None:
        dataset_path = args.dataset_path

    if args.dataset:
        meta = retrieve_meta()
        generate_dataset(meta, dataset_path, args.batch_size,
                         test_percentage=args.test_percentage)

    if args.load_model is None:
        model, graph, data_args, param_names, key_index = generate_model(
            args.batch_size, args.image_width, args.image_height, context=context, render=args.render_graph)
    else:
        model, graph, data_args, param_names, key_index = load_model(
            args.load_model, context=context)

    if args.train_model:
        model = train_model(model, graph, data_args, param_names, key_index, dataset_path, args.save_model,
                            args.batch_size, args.image_width, args.image_height,
                            iterations=args.learning_iterations,
                            learning_rate=args.learning_rate,
                            optimizer=args.optimizer)
