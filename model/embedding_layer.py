"""Implement of embedding layer"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mxnet import init
from mxnet.gluon import nn
from mxnet import nd


class EmbeddingSharedWeights(nn.Block):
    def __init__(self, vocab_size, hidden_size, **kwargs):
        super(EmbeddingSharedWeights, self).__init__(**kwargs)
        self.vocab_size = vocab_size
        self.hidden_size = hidden_size
        with self.name_scope():
            self.embedding = nn.Embedding(vocab_size, hidden_size,
                                          weight_initializer=init.Normal(sigma=self.hidden_size ** -0.5))

    def forward(self, x):
        return self.embedding(x)

    def linear(self, x):
        ctx = x.context
        with self.name_scope():
            batch_size = x.shape[0]
            length = x.shape[1]

            x = nd.reshape(x, (-1, self.hidden_size))
            logits = nd.dot(x, self.embedding.weight.data(ctx=ctx).T)

            return nd.reshape(logits, (batch_size, length, self.vocab_size))
