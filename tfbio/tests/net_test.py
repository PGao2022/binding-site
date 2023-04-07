import sys
import os
import tempfile

import re

import math
import numpy as np
import tensorflow as tf
from tensorflow.python import pywrap_tensorflow

import pytest

path = os.path.dirname(os.path.realpath(__file__))
sys.path.append(os.path.split(path)[0])


def teardown_function(function):
    tf.reset_default_graph()


@pytest.mark.parametrize('out_chnls', (8, 16), ids=lambda x: 'o=%s' % x)
@pytest.mark.parametrize('conv_patch', (2, 5, 10), ids=lambda x: 'c=%s' % x)
@pytest.mark.parametrize('pool_patch', (2, 3), ids=lambda x: 'p=%s' % x)
def test_hidden_conv3D(out_chnls, conv_patch, pool_patch):
    from tfbio.net import hidden_conv3D

    x = tf.placeholder(tf.float32, shape=(None, 21, 21, 21, 19))

    h = hidden_conv3D(x, out_chnls, conv_patch=conv_patch,
                      pool_patch=pool_patch, name='conv')

    input_tensor = h
    # there are 4 operations between x and h so we need 4 steps
    # to get back to x:
    # MaxPooling -> Add -> Conv -> x
    for _ in range(4):
        input_tensor = input_tensor.op.inputs[0]
    assert input_tensor == x
    shape = h.get_shape().as_list()
    s = math.ceil(21 / pool_patch)
    assert shape == [None, s, s, s, out_chnls]


@pytest.mark.parametrize('out_size', (8, 16), ids=lambda x: 'o=%s' % x)
def test_hidden_fcl(out_size):
    from tfbio.net import hidden_fcl

    x = tf.placeholder(tf.float32, shape=(None, 100))

    keep_prob = tf.placeholder(tf.float32)
    h = hidden_fcl(x, out_size, keep_prob, name='fc')
    input_tensor = h
    # there are 5 operations between x and h so we need 5 steps
    # to get back to x:
    # Dropout -> ReLU -> Add -> MatMul -> x
    for _ in range(5):
        input_tensor = input_tensor.op.inputs[0]
    assert input_tensor == x

    shape = h.get_shape().as_list()
    assert shape == [None, out_size]


@pytest.mark.parametrize('conv_patch', (2, 5, 10), ids=lambda x: 'c=%s' % x)
@pytest.mark.parametrize('pool_patch', (2, 3), ids=lambda x: 'p=%s' % x)
def test_convolve3D(conv_patch, pool_patch):
    from tfbio.net import hidden_conv3D, convolve3D

    out_chnls = [8, 16]

    g1 = tf.Graph()
    with g1.as_default():
        x = tf.placeholder(tf.float32, shape=(None, 21, 21, 21, 19))
        h11 = hidden_conv3D(x, out_chnls[0], conv_patch=conv_patch,
                            pool_patch=pool_patch, name='conv0')
        h12 = hidden_conv3D(h11, out_chnls[1], conv_patch=conv_patch,
                            pool_patch=pool_patch, name='conv1')
    def1 = g1.as_graph_def().SerializeToString()

    g2 = tf.Graph()
    with g2.as_default():
        x = tf.placeholder(tf.float32, shape=(None, 21, 21, 21, 19))
        h2 = convolve3D(x, out_chnls, conv_patch=conv_patch,
                        pool_patch=pool_patch)
    def2 = g2.as_graph_def().SerializeToString()

    # graphs should be identical
    assert not pywrap_tensorflow.EqualGraphDefWrapper(def1, def2)

    # check if list with single ement works
    g1 = tf.Graph()
    with g1.as_default():
        x = tf.placeholder(tf.float32, shape=(None, 21, 21, 21, 19))
        h11 = hidden_conv3D(x, out_chnls[0], conv_patch=conv_patch,
                            pool_patch=pool_patch, name='conv0')
    def1 = g1.as_graph_def().SerializeToString()

    g2 = tf.Graph()
    with g2.as_default():
        x = tf.placeholder(tf.float32, shape=(None, 21, 21, 21, 19))
        h1 = convolve3D(x, out_chnls[:1], conv_patch=conv_patch,
                        pool_patch=pool_patch)
    def2 = g2.as_graph_def().SerializeToString()
    # graphs should be identical
    assert not pywrap_tensorflow.EqualGraphDefWrapper(def1, def2)


def test_feedforward():
    from tfbio.net import hidden_fcl, feedforward

    out_sizes = [16, 8]

    g1 = tf.Graph()
    with g1.as_default():
        x = tf.placeholder(tf.float32, shape=(None, 100))
        kp = tf.constant(1.0)
        h11 = hidden_fcl(x, out_sizes[0], keep_prob=kp, name='fc0')
        h12 = hidden_fcl(h11, out_sizes[1], keep_prob=kp, name='fc1')
    def1 = g1.as_graph_def().SerializeToString()

    g2 = tf.Graph()
    with g2.as_default():
        x = tf.placeholder(tf.float32, shape=(None, 100))
        kp = tf.constant(1.0)
        h2 = feedforward(x, out_sizes, keep_prob=kp)
    def2 = g2.as_graph_def().SerializeToString()

    # graphs should be identical
    assert not pywrap_tensorflow.EqualGraphDefWrapper(def1, def2)

    # check if list with single ement works
    g1 = tf.Graph()
    with g1.as_default():
        x = tf.placeholder(tf.float32, shape=(None, 100))
        kp = tf.constant(1.0)
        h11 = hidden_fcl(x, out_sizes[0], keep_prob=kp, name='fc0')
    def1 = g1.as_graph_def().SerializeToString()

    g2 = tf.Graph()
    with g2.as_default():
        x = tf.placeholder(tf.float32, shape=(None, 100))
        kp = tf.constant(1.0)
        h1 = feedforward(x, out_sizes[:1], keep_prob=kp)
    def2 = g2.as_graph_def().SerializeToString()

    # graphs should be identical
    assert not pywrap_tensorflow.EqualGraphDefWrapper(def1, def2)


def test_make_SB_network():
    from tfbio.net import make_SB_network

    config = {'isize': 10, 'in_chnls': 8, 'osize': 1,
              'conv_patch': 3, 'pool_patch': 2, 'conv_channels': [4, 8, 56],
              'dense_sizes': [10, 50, 20]}

    graph = make_SB_network(**config)

    x = graph.get_tensor_by_name('input/structure:0')
    assert x.get_shape().as_list() == ([None] + [config['isize']] * 3
                                       + [config['in_chnls']])
    t = graph.get_tensor_by_name('input/affinity:0')
    assert t.get_shape().as_list() == [None, config['osize']]

    prev_size = config['in_chnls']
    for i, curr_size in enumerate(config['conv_channels']):
        w = graph.get_tensor_by_name('convolution/conv%s/w:0' % i)
        b = graph.get_tensor_by_name('convolution/conv%s/b:0' % i)
        assert w.get_shape().as_list() == ([config['conv_patch']] * 3
                                           + [prev_size, curr_size])
        assert b.get_shape().as_list() == [curr_size]
        prev_size = curr_size

    flat = graph.get_tensor_by_name('fully_connected/h_flat:0')
    prev_size = flat.get_shape().as_list()[1]

    for i, curr_size in enumerate(config['dense_sizes']):
        w = graph.get_tensor_by_name('fully_connected/fc%s/w:0' % i)
        b = graph.get_tensor_by_name('fully_connected/fc%s/b:0' % i)
        assert w.get_shape().as_list() == [prev_size, curr_size]
        assert b.get_shape().as_list() == [curr_size]
        prev_size = curr_size


@pytest.mark.parametrize('n_bins', (5, 50, 200), ids=lambda x: '%s bins' % x)
def test_custom_summary_histogram(n_bins):
    from tfbio.net import custom_summary_histogram

    values = np.random.randn(10000)

    histogram = custom_summary_histogram(values, num_bins=n_bins)
    assert isinstance(histogram, tf.HistogramProto)
    assert n_bins == len(histogram.bucket)
    assert np.allclose(histogram.min, values.min())
    assert np.allclose(histogram.max, values.max())
    assert np.allclose(histogram.sum, values.sum())
    assert np.allclose(histogram.sum_squares, (values ** 2).sum())
    assert histogram.num == 10000


@pytest.mark.parametrize('dpi', (100, 300), ids=lambda x: 'dpi=%s' % x)
@pytest.mark.parametrize('w', (2, 5), ids=lambda x: 'w=%s' % x)
@pytest.mark.parametrize('h', (5, 8), ids=lambda x: 'h=%s' % x)
def test_custom_summary_image(dpi, w, h):
    from tfbio.net import custom_summary_image
    import matplotlib.pyplot as plt

    fig, ax = plt.subplots(figsize=(w, h), dpi=dpi)
    ax.scatter(np.random.rand(100), np.random.rand(100))
    image = custom_summary_image(fig)
    plt.close(fig)

    assert isinstance(image, tf.Summary.Image)
    assert image.width == dpi * w
    assert image.height == dpi * h
    assert image.colorspace == 3


@pytest.mark.parametrize('num_features', [5, 10, 15],
                         ids=lambda x: '%s features' % x)
def test_feature_importance_plot(num_features):
    from tfbio.net import feature_importance_plot

    image = feature_importance_plot(
        np.random.rand(num_features),
        labels=['label %s' % i for i in range(num_features)])
    assert isinstance(image, tf.Summary.Image)
    image = feature_importance_plot(np.random.rand(num_features))
    assert isinstance(image, tf.Summary.Image)


@pytest.mark.parametrize('args, err, msg', (
    ([1], TypeError, 'values must be an array'),
    ([np.array(['a', 'b', 'c'])], ValueError, 'values'),
    ([np.arange(10), 10.0], TypeError, 'num_bins must be int'),
    ([np.arange(10), -5], ValueError, 'num_bins must be positive')
), ids=('single int', 'wrong values', 'float bins', 'negative bins'))
def test_wrong_histogram(args, err, msg):
    from tfbio.net import custom_summary_histogram

    with pytest.raises(err, match=msg):
        custom_summary_histogram(*args)


@pytest.mark.parametrize('args, err, msg', (
    ([1], TypeError, 'values must be a 1D sequence'),
    ([np.array(['a', 'b', 'c'])], ValueError, 'values must be a 1D sequence'),
    ([[[1], [1]]], ValueError, 'values must be a 1D sequence'),
    ([[10], 'a'], TypeError, 'labels must br a 1D sequence'),
    ([[5, 10], ['a']], ValueError, 'values and labels must have equal lengths'),
), ids=('single int', 'wrong values', 'wrong values shape',
        'wrong labels type', 'wrong labels len'))
def test_wrong_feature_importance_plot(args, err, msg):
    from tfbio.net import feature_importance_plot
    with pytest.raises(err):
        feature_importance_plot(*args)


@pytest.mark.parametrize('num_features', [5, 10, 15],
                         ids=lambda x: '%s features' % x)
def test_make_summaries_SB(num_features):
    from tfbio.net import make_SB_network, make_summaries_SB

    g = make_SB_network(in_chnls=num_features)
    labels = ['label %s' % i for i in range(num_features)]
    net_summ, training_summ = make_summaries_SB(g, feature_labels=labels)
    assert isinstance(net_summ, tf.Tensor)
    assert net_summ.dtype == tf.string
    assert isinstance(training_summ, tf.Tensor)
    assert training_summ.dtype == tf.string
    assert net_summ.graph == g
    assert training_summ.graph == g


def test_summary_writer():
    from tfbio.net import SummaryWriter

    with tempfile.TemporaryDirectory() as tempdir:
        summary_dir = os.path.join(tempdir, 'test')
        with SummaryWriter(summary_dir) as writer:
            for i in range(10):
                summary = tf.Summary()
                summary.value.add(tag='my_summary', simple_value=i)
                writer.add_summary(summary, i)

        # this should produce single file with events in tempdir/test/
        assert os.path.isdir(summary_dir)
        event_files = os.listdir(summary_dir)
        assert len(event_files) == 1
        assert re.match('events\.out\.tfevents\.[0-9]+\.\w', event_files[0])
