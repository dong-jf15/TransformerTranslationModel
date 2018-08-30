"""Translate text or files using trained transformer model."""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import xrange
import mxnet as mx
from mxnet import nd

import utils.tokenizer as tokenizer
import numpy as np

_DECODE_BATCH_SIZE = 32
_EXTRA_DECODE_LENGTH = 100
_BEAM_SIZE = 4
_ALPHA = 0.6
ctx = mx.gpu()


def _get_sorted_inputs(filename):
    """Read and sort lines from the file sorted by decreasing length.

    Args:
      filename: String name of file to read inputs from.
    Returns:
      Sorted list of inputs, and dictionary mapping original index->sorted index
      of each element.
    """
    with open(filename) as f:
        records = f.read().split("\n")
        inputs = [record.strip() for record in records]
        if not inputs[-1]:
            inputs.pop()

    input_lens = [(i, len(line.split())) for i, line in enumerate(inputs)]
    sorted_input_lens = sorted(input_lens, key=lambda x: x[1], reverse=True)

    sorted_inputs = []
    sorted_keys = {}
    for i, (index, _) in enumerate(sorted_input_lens):
        sorted_inputs.append(inputs[index])
        sorted_keys[index] = i
    return sorted_inputs, sorted_keys


def _encode_and_add_eos(line, subtokenizer):

    """Encode line with subtokenizer, and add EOS id to the end."""

    return subtokenizer.encode(line) + [tokenizer.EOS_ID]


def _trim_and_decode(ids, subtokenizer):
    """Trim EOS and PAD tokens from ids, and decode to return a string."""
    try:
        index = list(ids).index(tokenizer.EOS_ID)
        return subtokenizer.decode(ids[:index])
    except ValueError:  # No EOS found in sequence
        return subtokenizer.decode(ids)


def translate_file(model, subtokenizer, input_file, output_file=None, print_all_translations=True):
    """Translate lines in file, and save to output file if specified.

      Args:
        estimator: tf.Estimator used to generate the translations.
        subtokenizer: Subtokenizer object for encoding and decoding source and
           translated lines.
        input_file: file containing lines to translate
        output_file: file that stores the generated translations.
        print_all_translations: If true, all translations are printed to stdout.

      Raises:
        ValueError: if output file is invalid.
      """
    print("Begin translate file from: %s" % input_file)
    batch_size = _DECODE_BATCH_SIZE

    sorted_inputs, sorted_keys = _get_sorted_inputs(input_file)
    num_decode_batches = (len(sorted_inputs) - 1) // batch_size + 1

    def get_batch(idx):
        if idx == (num_decode_batches - 1):
            ret = sorted_inputs[idx * batch_size:-1]
            leng = len(ret)
        else:
            ret = sorted_inputs[idx * batch_size:idx * batch_size + batch_size]
            leng = len(ret)

        max_length = 0
        for j in xrange(leng):
            ret[j] = _encode_and_add_eos(ret[j], subtokenizer)
            if max_length < len(ret[j]):
                max_length = len(ret[j])

        for k in xrange(leng):
            ret[k] = ret[k] + np.zeros(max_length - len(ret[k])).tolist()

        return nd.array(ret, ctx=ctx)

    translations = []
    for i in xrange(num_decode_batches):
        print("\t Tranlate batch %d of %d" % (i, num_decode_batches))
        output = model(get_batch(i))
        output = output['outputs']
        output = nd.cast(output, dtype='int32')
        for j in xrange(len(output)):
            translation = _trim_and_decode(output[j].asnumpy().tolist(), subtokenizer)
            translations.append(translation)

    with open(output_file) as f:
        print("Finished translation and write the translated file.")
        for index in xrange(len(sorted_keys)):
            f.write("%s\n" % translations[sorted_keys[index]])
        f.close()




