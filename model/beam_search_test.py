from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import mxnet as mx
from mxnet import nd
import numpy as np
import unittest

import beam_search
ctx = mx.cpu()


class BeamSearchHelperTests(unittest.TestCase):

  def test_expand_to_beam_size(self):
    x = nd.ones([7, 4, 2, 5], ctx=ctx)
    x = beam_search._expand_to_beam_size(x, 3)
    self.assertEqual((7, 3, 4, 2, 5), x.shape)

  def test_flatten_beam_dim(self):
    x = nd.ones([7, 4, 2, 5], ctx=ctx)
    x = beam_search._flatten_beam_dim(x)
    self.assertEqual((28, 2, 5), x.shape)

  def test_unflatten_beam_dim(self):
    x = nd.ones([28, 2, 5], ctx=ctx)
    x = beam_search._unflatten_beam_dim(x, 7, 4)
    self.assertEqual((7, 4, 2, 5), x.shape)

  def test_gather_beams(self):
    x = nd.reshape(nd.arange(24, ctx=ctx), (2, 3, 4))
    # x looks like:  [[[ 0  1  2  3]
    #                  [ 4  5  6  7]
    #                  [ 8  9 10 11]]
    #
    #                 [[12 13 14 15]
    #                  [16 17 18 19]
    #                  [20 21 22 23]]]

    y = beam_search._gather_beams([x], nd.array([[1, 2], [0, 2]], ctx=ctx, dtype='int32'), 2, 2)

    self.assertEqual([[[4, 5, 6, 7],
                          [8, 9, 10, 11]],
                         [[12, 13, 14, 15],
                          [20, 21, 22, 23]]],
                        y[0].asnumpy().tolist())

  def test_gather_topk_beams(self):
    x = nd.reshape(nd.arange(24, ctx=ctx), (2, 3, 4))
    x_scores = nd.array([[0, 1, 1], [1, 0, 1]], ctx=ctx)

    y = beam_search._gather_topk_beams([x], x_scores, 2, 2)

    self.assertEqual([[[4, 5, 6, 7],
                        [8, 9, 10, 11]],
                         [[12, 13, 14, 15],
                          [20, 21, 22, 23]]],
                        y[0].asnumpy().tolist())


if __name__ == "__main__":
    unittest.main()