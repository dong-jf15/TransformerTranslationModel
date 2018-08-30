from __future__ import absolute_import
from __future__ import division
from __future__ import print_function
import math

import mxnet as mx
from mxnet import nd
from six.moves import xrange

_NEG_INF = -1e9


def get_position_encoding(length, hidden_size, ctx, min_timescale=1.0, max_timescale=1.0e4):
    """Return positional encoding.

    Calculates the position encoding as a mix of sine and cosine functions with
    geometrically increasing wavelengths.
    Defined and formulized in Attention is All You Need, section 3.5.

    Args:
      length: Sequence length.
      hidden_size: Size of the
      min_timescale: Minimum scale that will be applied at each position
      max_timescale: Maximum scale that will be applied at each position

    Returns:
      Tensor with shape [length, hidden_size]
    """
    position = nd.arange(length, ctx=ctx)
    num_timescales = hidden_size // 2
    log_timescale_increment = (math.log(float(max_timescale) / float(min_timescale)) / (float(num_timescales) - 1))
    inv_timescales = min_timescale * nd.exp(nd.arange(num_timescales, ctx=ctx) * -log_timescale_increment)
    scaled_time = nd.expand_dims(position, 1) * nd.expand_dims(inv_timescales, 0)
    signal = nd.concat(nd.sin(scaled_time), nd.cos(scaled_time), dim=1)
    return signal


def get_decoder_self_attention_bias(length, ctx):
    """Calculate bias for decoder that maintains model's autoregressive property.

     Creates a tensor that masks out locations that correspond to illegal
     connections, so prediction at position i cannot draw information from future
     positions.

     Args:
       length: int length of sequences in batch.

     Returns:
       float tensor of shape [1, 1, length, length]
     """
    valid_locs = low_triangle(length, ctx)
    valid_locs = valid_locs.reshape((1, 1, length, length))
    decoder_bias = _NEG_INF * (1.0 - valid_locs)
    temp = nd.zeros(shape=(1, 1, length, 1))

    return nd.concat(temp, decoder_bias, dim=3)


def low_triangle(length, ctx):
    matrix = nd.ones((length, length), ctx=ctx)
    for i in xrange(length):
        for j in xrange(length):
            matrix[i, j] = (i >= j)

    return matrix


def get_padding(x, padding_value=0):
    return nd.equal(x, padding_value)


def get_padding_bias(x):
    padding = get_padding(x)
    attention_bias = padding * _NEG_INF
    attention_bias = nd.expand_dims(nd.expand_dims(attention_bias, 1), 1)
    return attention_bias

