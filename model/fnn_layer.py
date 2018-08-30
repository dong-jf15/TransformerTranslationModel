"""Implement of fnn layer"""
from __future__ import division
from __future__ import print_function
from __future__ import absolute_import


from mxnet.gluon import nn
from mxnet import nd
import numpy as np
import mxnet as mx


class FeedForwardNetwork(nn.Block):
    """Fully connected feedforward network"""

    def __init__(self, hidden_size, filter_size, relu_dropout, train, **kwargs):
        super(FeedForwardNetwork, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.filter_size = filter_size
        self.relu_dropout = relu_dropout
        self.train = train

        with self.name_scope():
            self.filter_dense_layer = nn.Dense(self.filter_size, activation='relu', use_bias=True, flatten=False)
            self.output_dense_layer = nn.Dense(self.hidden_size, use_bias=True, flatten=False)
            self.dropout = nn.Dropout(1.0 - self.relu_dropout)

    def forward(self, x, padding=None):
        ctx = x.context
        batch_size = x.shape[0]
        length = x.shape[1]
        if padding is not None:
            # Flattten padding to [batch_size * length]
            pad_mask = nd.reshape(padding, (-1))
            nonpad_ids = nd.array(np.where(pad_mask.asnumpy() < 1e-9), ctx=ctx)

            # Reshape x to [batch_size*length, hidden_size] to remove padding
            x = nd.reshape(x, (-1, self.hidden_size))
            x = nd.gather_nd(x, indices=nonpad_ids)

            # Reshape x from 2 dimensions to 3 dimensions
            x = nd.expand_dims(x, axis=0)

        output = self.filter_dense_layer(x)
        if self.train:
            output = self.dropout(output)
        output = self.output_dense_layer(output)

        if padding is not None:
            output = nd.squeeze(output, axis=0)
            output = nd.scatter_nd(data=output, indices=nonpad_ids, shape=(batch_size * length, self.hidden_size))
            output = nd.reshape(output, shape=(batch_size, length, self.hidden_size))

        return output


