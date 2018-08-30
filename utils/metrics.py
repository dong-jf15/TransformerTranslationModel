"""
Function for calculating loss, accuracy, and other model metrics
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import collections
import math

import numpy as np
from six.moves import xrange
import mxnet as mx
from mxnet import nd


def _pad_tensors_to_same_length(x, y):
    """Pad x and y so that the result have the same length (second dimension)"""
    x_length = x.shape[1]
    y_length = y.shape[1]

    max_length = max(x_length, y_length)
    x = nd.expand_dims(x, axis=0)
    x = nd.pad(x, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, 0, max_length - x_length, 0, 0))
    x = nd.squeeze(x, axis=0)

    y = nd.expand_dims(y, axis=0)
    y = nd.expand_dims(y, axis=0)
    y = nd.pad(y, mode="constant", constant_value=0, pad_width=(0, 0, 0, 0, 0, 0, 0, max_length - y_length))
    y = nd.squeeze(y, axis=0)
    y = nd.squeeze(y, axis=0)

    return x, y


def padded_cross_entropy_loss(logits, labels, smoothing, vocab_size):
    """
    Calculate cross entropy loss while ignoring padding.

    :param logits: Tensor of size [batch_size, length_logits, vocab_size]
    :param labels: Tensor of size [batch_size, length_labels]
    :param smoothing: Label smoothing constant, used to determine the on an off values
    :param vocab_size: int size of the vocabulary
    :return: a float32 tennsor with shape
    [batch_size, max(length_logits, length_labels)]
    """
    logits, labels = _pad_tensors_to_same_length(logits, labels)

    confidence = 1.0 - smoothing
    low_confidence = (1.0 - confidence) / float(vocab_size - 1)
    soft_targets = nd.one_hot(
        indices=nd.cast(labels, dtype='int32'),
        depth=vocab_size,
        on_value=confidence,
        off_value=low_confidence
    )
    softmax_cross_entropy = mx.gluon.loss.SoftmaxCrossEntropyLoss(
        axis=-1,
        sparse_label=False,
        from_logits=True
    )
    xentropy = softmax_cross_entropy(logits, soft_targets)

    normalizing_constant = -(confidence * np.log(confidence) + float(vocab_size - 1)
                             * low_confidence * np.log(low_confidence + 1e-20))
    xentropy = xentropy - normalizing_constant

    return xentropy


def _get_ngrams_with_counter(segment, max_order):
    """
    Extracts all n-grams up to a given maximum order from an input segment.

    Args:
    segment: text segment from which n-grams will be extracted.
    max_order: maximum length in tokens of the n-grams returned by this
        methods.

    Returns:
        The Counter containing all n-grams upto max_order in segment
        with a count of how many times each n-gram occurred.
    """
    ngram_counts = collections.Counter()
    for order in xrange(1, max_order + 1):
        for i in xrange(0, len(segment) - order + 1):
            ngram = tuple(segment[i:i + order])
            ngram_counts[ngram] = ngram_counts[ngram] + 1
    return ngram_counts


def compute_bleu(reference_corpus, translation_corpus, max_order=4, use_bp=True):
    """Computes BLEU score of translated segments against one or more references.

    Args:
      reference_corpus: list of references for each translation. Each
          reference should be tokenized into a list of tokens.
      translation_corpus: list of translations to score. Each translation
          should be tokenized into a list of tokens.
      max_order: Maximum n-gram order to use when computing BLEU score.
      use_bp: boolean, whether to apply brevity penalty.

    Returns:
      BLEU score.
    """
    reference_length = 0
    translation_length = 0
    bp = 1.0
    geo_mean = 0

    matches_by_order = [0] * max_order
    possible_matches_by_order = [0] * max_order
    precisions = []

    for (references, translations) in zip(reference_corpus, translation_corpus):
        reference_length = reference_length + len(references)
        translation_length = translation_length + len(translations)
        ref_ngram_counts = _get_ngrams_with_counter(references, max_order)
        translation_ngram_counts = _get_ngrams_with_counter(translations, max_order)

        overlap = dict((ngram,
                        min(count, translation_ngram_counts[ngram]))
                       for ngram, count in ref_ngram_counts.items())

        for ngram in overlap:
            matches_by_order[len(ngram) - 1] = matches_by_order[len(ngram) - 1] + overlap[ngram]
        for ngram in translation_ngram_counts:
            possible_matches_by_order[len(ngram) - 1] = possible_matches_by_order[len(ngram) - 1] + \
                                                        translation_ngram_counts[ngram]

    precisions = [0] * max_order
    smooth = 1.0

    for i in xrange(0, max_order):
        if possible_matches_by_order[i] > 0:
            precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[i]
            if matches_by_order[i] > 0:
                precisions[i] = float(matches_by_order[i]) / possible_matches_by_order[
                    i]
            else:
                smooth = smooth * 2
                precisions[i] = 1.0 / (smooth * possible_matches_by_order[i])
        else:
            precisions[i] = 0.0

    if max(precisions) > 0:
        p_log_sum = sum(math.log(p) for p in precisions if p)
        geo_mean = math.exp(p_log_sum / max_order)

    if use_bp:
        ratio = translation_length / reference_length
        bp = math.exp(1 - 1. / ratio) if ratio < 1.0 else 1.0
    bleu = geo_mean * bp
    return np.float32(bleu)
