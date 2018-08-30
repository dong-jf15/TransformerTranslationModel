from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from utils.tokenizer import *
from mxnet.gluon.data import dataset
from six.moves import xrange
import mxnet as mx
import mxnet.ndarray as nd
import random

_FILE_SHUFFLE_BUFFER = 100
_READ_RECORD_BUFFER = 8 * 1000 * 1000
_MIN_BOUNDARY = 8
_BOUNDARY_SCALE = 1.1


class TranslationDataset(dataset.Dataset):
    """
    A dataset which load sentences from files and decode them to int arrays.
    """

    def __init__(self, dir_lang1, dir_lang2, subtokenizer):
        self._dir_lang1 = dir_lang1
        self._dir_lang2 = dir_lang2
        self._list = []
        self._subtokenizer = subtokenizer
        self._current_idx = 0

        with open(dir_lang1) as f:
            self._lang1 = f.readlines()
            f.close()
        with open(dir_lang2) as f:
            self._lang2 = f.readlines()
            f.close()
        for i in xrange(len(self._lang1)):
            self._list.append({'input': self._subtokenizer.encode(self._lang1[i], add_eos=True),
                               'targets': self._subtokenizer.encode(self._lang2[i], add_eos=True)})

    def __getitem__(self, idx):
        if idx >= len(self._list):
            print('Cant find the item, idx too big')
        if idx < 0:
            print('Cant find the item, idx less than zero')
        return self._list[idx]

    def __len__(self):
        return len(self._list)

    def get_mini_batch(self, batch_size=10):
        """Get a mini batch for training"""

        if (self._current_idx + batch_size) > len(self._list):
            random.shuffle(self._list)
            self._current_idx = 0
        max_length_input = 0
        max_length_targets = 0
        for i in xrange(0, batch_size, 1):
            if len(self._list[self._current_idx+i]['input']) > max_length_input:
                max_length_input = len(self._list[self._current_idx+i]['input'])
            if len(self._list[self._current_idx+i]['targets']) > max_length_targets:
                max_length_targets = len(self._list[self._current_idx+i]['targets'])

        input = []

        targets = []

        for i in xrange(0, batch_size, 1):
            length_input = len(self._list[self._current_idx + i]['input'])

            add_input = np.append(np.array(self._list[self._current_idx + i]['input']),
                                  np.zeros(max_length_input - length_input))

            input.append(add_input)

            length_targets = len(self._list[self._current_idx + i]['targets'])

            add_targets = np.append(np.array(self._list[self._current_idx + i]['targets']),
                                    np.zeros(max_length_targets - length_targets))

            targets.append(add_targets)

        self._current_idx = self._current_idx + batch_size

        return {'input': nd.array(input), 'targets': nd.array(targets)}
