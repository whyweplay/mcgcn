#!/usr/local/bin/python
from inits import *
import tensorflow as tf
import numpy as np
import scipy.sparse as sp
flags = tf.app.flags
FLAGS = flags.FLAGS


_LAYER_UIDS = {}


def get_layer_uid(layer_name=''):
    if layer_name not in _LAYER_UIDS:
        _LAYER_UIDS[layer_name] = 1
        return 1
    else:
        _LAYER_UIDS[layer_name] += 1
        return _LAYER_UIDS[layer_name]


def sparse_dropout(x, keep_prob, noise_shape):
    random_tensor = keep_prob
    random_tensor += tf.random_uniform(noise_shape)
    dropout_mask = tf.cast(tf.floor(random_tensor), dtype=tf.bool)
    pre_out = tf.sparse_retain(x, dropout_mask)
    return pre_out * (1./keep_prob)


def dot(x, y, sparse=False):
    if sparse:
        res = tf.sparse_tensor_dense_matmul(x, y)

    else:
        res = tf.matmul(x, y)
    return res

def dot_geo(x, y):
    return tf.multiply(x, y)




class Layer(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            layer = self.__class__.__name__.lower()
            name = layer + '_' + str(get_layer_uid(layer))
        self.name = name
        self.vars = {}
        logging = kwargs.get('logging', False)
        self.logging = logging
        self.sparse_inputs = False

    def _call(self, inputs, pattern_dis, tuopu):
        return inputs

    def __call__(self, inputs, pattern_dis, tuopu):
        with tf.name_scope(self.name):
            if self.logging and not self.sparse_inputs:
                tf.summary.histogram(self.name + '/inputs', inputs)
            outputs = self._call(inputs, pattern_dis, tuopu)
            if self.logging:
                tf.summary.histogram(self.name + '/outputs', outputs)
            return outputs

    def _log_vars(self):
        for var in self.vars:
            tf.summary.histogram(self.name + '/vars/' + var, self.vars[var])

class MCGraphConvolution(Layer):
    def __init__(self, input_dim, output_dim, placeholders, dropout=0.,
                 sparse_inputs=False, act=tf.nn.relu, bias=False, **kwargs):
        super(MCGraphConvolution, self).__init__(**kwargs)

        if dropout:
            self.dropout = placeholders['dropout']
        else:
            self.dropout = 0.0

        self.act = act
        self.support = placeholders['support']
        self.sparse_inputs = sparse_inputs
        self.bias = bias


        self.num_features_nonzero = placeholders['num_features_nonzero']

        with tf.variable_scope(self.name + '_vars'):
            self.vars['weights_'] = glorot([input_dim, output_dim], name='weights_')
            if self.bias:
                self.vars['bias'] = zeros([output_dim], name='bias')


        if self.logging:
            self._log_vars()

    def _call(self, inputs, pattern_dis, tuopu_adj):
        x = inputs
        pattern_dis = pattern_dis
        tuopu = tuopu_adj



        if self.sparse_inputs:
            x = sparse_dropout(x, 1-self.dropout, self.num_features_nonzero)

        else:
            x = tf.nn.dropout(x, 1-self.dropout)


        pre_sup = dot(x, self.vars['weights_'], sparse=self.sparse_inputs)
        out_gcn = dot(tuopu, pre_sup, sparse = False)
        sp_ordered = tf.sparse_reorder(self.support[1])
        support_dis = tf.sparse_tensor_to_dense(sp_ordered)
        cal_dis = dot_geo(pattern_dis, support_dis)
        out_geo_gcn = dot(cal_dis, pre_sup, sparse = False)


        output = (1 - FLAGS.alpha) * out_gcn + FLAGS.alpha * out_geo_gcn



        if self.bias:
            output += self.vars['bias']

        return self.act(output)
