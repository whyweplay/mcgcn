#!/usr/local/bin/python
from layers import *
from metrics import *

flags = tf.app.flags
FLAGS = flags.FLAGS


class Model(object):
    def __init__(self, **kwargs):
        allowed_kwargs = {'name', 'logging'}
        for kwarg in kwargs.keys():
            assert kwarg in allowed_kwargs, 'Invalid keyword argument: ' + kwarg
        name = kwargs.get('name')
        if not name:
            name = self.__class__.__name__.lower()
        self.name = name

        logging = kwargs.get('logging', False)
        self.logging = logging

        self.vars = {}
        self.placeholders = {}

        self.layers = []
        self.activations = []
        self.activations_2 = []
        self.activations_3 = []

        self.inputs = None
        self.pattern_dis = None
        self.tuopu_adj = None
        self.outputs = None

        self.loss = 0
        self.cross_entropy_loss = 0
        self.accuracy = 0
        self.optimizer = None
        self.opt_op = None

    def _build(self):
        raise NotImplementedError

    def build(self):
        with tf.variable_scope(self.name):
            self._build()

        self.activations.append(self.inputs)
        self.activations_2.append(self.pattern_dis)
        self.activations_3.append(self.tuopu_adj)



        for layer in self.layers:
            hidden = layer(self.activations[-1], self.activations_2[-1], self.activations_3[-1])
            self.activations.append(hidden)
        self.outputs = self.activations[-1]



        variables = tf.get_collection(tf.GraphKeys.GLOBAL_VARIABLES, scope=self.name)
        self.vars = {var.name: var for var in variables}

        # Build metrics
        self._loss()
        self._accuracy()

        self.opt_op = self.optimizer.minimize(self.loss)

    def predict(self):
        pass

    def _loss(self):
        raise NotImplementedError

    def _accuracy(self):
        raise NotImplementedError

class MCGCN(Model):
    def __init__(self, placeholders, input_dim, **kwargs):
        super(MCGCN, self).__init__(**kwargs)

        self.inputs = placeholders['features']
        self.pattern_dis = placeholders['pattern_dis']
        self.tuopu_adj = placeholders['tuopu_adj']
        self.input_dim = input_dim
        self.output_dim = 1
        self.placeholders = placeholders


        self.optimizer = tf.train.AdamOptimizer(learning_rate=FLAGS.learning_rate)

        self.build()

    def _loss(self):
        # Weight decay loss

        for var in self.layers[0].vars.values():
            self.loss += FLAGS.weight_decay * tf.nn.l2_loss(var)


        self.loss += mse_loss(self.outputs, self.placeholders['labels'])


        self.cross_entropy_loss += rmse_loss(self.outputs, self.placeholders['labels'])

    def _accuracy(self):

        self.accuracy = mae_loss(self.outputs, self.placeholders['labels'])

    def _build(self):

        self.layers.append(MCGraphConvolution(
                                            input_dim=self.input_dim,
                                            output_dim=self.output_dim,
                                            placeholders=self.placeholders,
                                            act=lambda x: x,
                                            dropout=True,
                                            sparse_inputs=True,
                                            logging=self.logging))

