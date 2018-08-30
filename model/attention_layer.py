"""Implementation of multiheaded attention and self-attention layers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
from mxnet.gluon import nn
from six.moves import xrange
import mxnet as mx


class Attention(nn.HybridBlock):
    """Multi-headed attention layer."""

    def __init__(self, hidden_size, num_heads, attention_dropout, train, **kwargs):
        if hidden_size % num_heads != 0:
            raise ValueError("Hidden size must be evenly divisible by the number of "
                             "heads.")
        super(Attention, self).__init__(**kwargs)
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.attention_dropout = attention_dropout
        self.train = train

        with self.name_scope():
            self.q_dense_layer = nn.Dense(hidden_size, use_bias=False, flatten=False)
            self.k_dense_layer = nn.Dense(hidden_size, use_bias=False, flatten=False)
            self.v_dense_layer = nn.Dense(hidden_size, use_bias=False, flatten=False)

            self.output_dense_layer = nn.Dense(hidden_size, use_bias=False, flatten=False)
            self.Dropout_layer = nn.Dropout(1.0 - self.attention_dropout)

    def hybrid_forward(self, F, x, y, bias):
        """Apply attention mechanism to x and y.

            Args:
              x: a tensor with shape [batch_size, length_x, hidden_size]
              y: a tensor with shape [batch_size, length_y, hidden_size]
              bias: attention bias that will be added to the result of the dot product.
              cache: (Used during prediction) dictionary with tensors containing results
                of previous attentions. The dictionary must have the items:
                    {"k": tensor with shape [batch_size, i, key_channels],
                     "v": tensor with shape [batch_size, i, value_channels]}
                where i is the current decoded length.

            Returns:
              Attention layer output with shape [batch_size, length_x, hidden_size]
            """

        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        q = self.split_heads(F, q)
        k = self.split_heads(F, k)
        v = self.split_heads(F, v)

        depth = (self.hidden_size // self.num_heads)
        q = q * (depth ** -0.5)

        logits = my_matmul(F, q, F.transpose(k, axes=(0, 1, 3, 2)))
        logits = logits + bias
        weights = F.softmax(logits)
        if self.train:
            weights = self.Dropout_layer(weights)
        attention_output = my_matmul(F, weights, v)

        attention_output = self.combine_heads(F, attention_output)
        attention_output = self.output_dense_layer(attention_output)

        return attention_output

    def split_heads(self, F, x):
        """Split x into different heads, and transpose the resulting value.

            The tensor is transposed to insure the inner dimensions hold the correct
            values during the matrix multiplication.

            Args:
              x: A tensor with shape [batch_size, length, hidden_size]

            Returns:
              A tensor with shape [batch_size, num_heads, length, hidden_size/num_heads]
            """
        depth = (self.hidden_size // self.num_heads)

        x = F.reshape(x, shape=(0, 0, self.num_heads, depth))

        return F.transpose(x, axes=(0, 2, 1, 3))

    def combine_heads(self, F, x):
        """Combine tensor that has been split.

            Args:
              x: A tensor [batch_size, num_heads, length, hidden_size/num_heads]

            Returns:
              A tensor with shape [batch_size, length, hidden_size]
            """
        x = F.transpose(x, axes=(0, 2, 1, 3))
        return F.reshape(x, shape=(0, 0, self.hidden_size))


class SelfAttention(Attention):
    def hybrid_forward(self, F, x, bias):
        return super(SelfAttention, self).hybrid_forward(F, x, x, bias)


def my_matmul(F, x, y):

    state = [x, y, F.batch_dot(x[0], y[0]), F.ones(1)]

    def mul(data1, data2, out, index):
        temp = F.batch_dot(data1[index], data2[index])
        temp = F.concat(out, temp, dim=0)
        return _, [data1, data2, temp, index + 1]

    def mul_cond(data1, data2, out, index):
        return index < F.cast(F.shape_array(data1)[0], dtype='float32')

    _, ret = F.contrib.while_loop(mul_cond, mul, state, max_iterations=100)

    return ret


class SelfAttentionWithCache(Attention):
    def hybrid_forward(self, F, x, y, bias, cache):
        q = self.q_dense_layer(x)
        k = self.k_dense_layer(y)
        v = self.v_dense_layer(y)

        cache[0] = F.concat(cache[0], k, dim=1)
        cache[1] = F.concat(cache[1], v, dim=1)
        k = cache[0]
        v = cache[1]
        q = self.split_heads(F, q)
        k = self.split_heads(F, k)
        v = self.split_heads(F, v)

        depth = (self.hidden_size // self.num_heads)
        q = q * (depth ** -0.5)

        logits = my_matmul(F, q, F.transpose(k, axes=(0, 1, 3, 2)))
        logits = logits + bias
        weights = F.softmax(logits)
        if self.train:
            weights = self.Dropout_layer(weights)
        attention_output = my_matmul(F, weights, v)

        attention_output = self.combine_heads(F, attention_output)
        attention_output = self.output_dense_layer(attention_output)

        return attention_output


if __name__ == "__main__":
    net = SelfAttentionWithCache(16, 4, 0.1, 1)
    net.collect_params().initialize()
    cache = [mx.nd.zeros((4, 1, 16)), mx.nd.zeros((4, 1, 16))]
    a = mx.nd.ones((4, 5, 16))
    b = mx.nd.ones((4, 5, 16))
    bias = mx.nd.concat(mx.nd.zeros((1, 1, 5, 1)), mx.nd.ones((1, 1, 5, 5)), dim=3)
    print(net(a, b, bias, cache))
