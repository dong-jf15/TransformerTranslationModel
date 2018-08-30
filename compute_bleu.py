"""
Script to compute official BLEU score.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import argparse
import re
import sys
import unicodedata

import six

import utils.metrics as metrics
import transformer_main.translate_and_compute_bleu as translate_and_compute_bleu
import model.transformer as transformer
import model.model_params as model_params


class UnicodeRegex(object):
    """Ad-hoc hack to recognize all punctuation and symbols"""

    def __init__(self):
        punctuation = self.property_chars("P")
        self.nondigit_punct_re = re.compile(r"([^\d])([" + punctuation + r"])")
        self.punct_nondigit_re = re.compile(r"([" + punctuation + r"])([^\d])")
        self.symbol_re = re.compile("([" + self.property_chars("S") + "])")

    def property_chars(self, prefix):
        return "".join(six.unichr(x) for x in range(sys.maxunicode)
                       if unicodedata.category(six.unichr(x)).startswith(prefix))


uregex = UnicodeRegex()


def bleu_tokenize(string):
    r"""Tokenize a string following the official BLEU implementation.

    See https://github.com/moses-smt/mosesdecoder/'
             'blob/master/scripts/generic/mteval-v14.pl#L954-L983
    In our case, the input string is expected to be just one line
    and no HTML entities de-escaping is needed.
    So we just tokenize on punctuation and symbols,
    except when a punctuation is preceded and followed by a digit
    (e.g. a comma/dot as a thousand/decimal separator).

    Note that a numer (e.g. a year) followed by a dot at the end of sentence
    is NOT tokenized,
    i.e. the dot stays with the number because `s/(\p{P})(\P{N})/ $1 $2/g`
    does not match this case (unless we add a space after each sentence).
    However, this error is already in the original mteval-v14.pl
    and we want to be consistent with it.

    Args:
      string: the input string

    Returns:
      a list of tokens
    """
    string = uregex.nondigit_punct_re.sub(r"\1 \2 ", string)
    string = uregex.punct_nondigit_re.sub(r" \1 \2", string)
    string = uregex.symbol_re.sub(r" \1 ", string)
    return string.split()


def bleu_wrapper(ref_filename, hyp_filename, case_sensitive=False):
    """compute BLEU for two files."""
    print("Compute BLEU score between two files.")
    with open(ref_filename) as f1:
        ref_lines = f1.read().strip().splitlines()
    with open(hyp_filename) as f2:
        hyp_lines = f2.read().strip().splitlines()

    if len(ref_lines) != len(hyp_lines):
        raise ValueError("Reference and translation files have diffenrent number of lines")

    if not case_sensitive:
        ref_lines = [x.lower() for x in ref_lines]
        hyp_lines = [x.lower() for x in hyp_lines]

    ref_tokens = [bleu_tokenize(x) for x in ref_lines]
    hyp_tokens = [bleu_tokenize(x) for x in hyp_lines]

    return metrics.compute_bleu(ref_tokens, hyp_tokens) * 100

	

