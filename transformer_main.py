"""
define the train and eval loop
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tempfile
import random
import numpy.random
import os
import gc

from six.moves import xrange
import mxnet as mx
from mxnet import gluon, init, autograd

import compute_bleu
import model.transformer as transformer
import model.model_params as model_params
import translate
import utils.dataset as dataset
import utils.metrics as metrics
import utils.tokenizer as tokenizer

import numpy as np
from time import time

INF = 10000
num_gpu = 4
ctx = [mx.gpu(i) for i in range(num_gpu)]


def get_learning_rate(learning_rate, hidden_size, learning_rate_warmup_steps, global_step):
    warmup_steps = float(learning_rate_warmup_steps)
    step = float(global_step)

    learning_rate = (hidden_size ** -0.5) * learning_rate

    learning_rate = np.minimum(1.0, step / warmup_steps) * learning_rate

    learning_rate = (np.maximum(step, warmup_steps) ** -0.5) * learning_rate

    return learning_rate


def translate_and_compute_bleu(model, subtokenizer, bleu_source, bleu_ref):
    """
    Translate file and report the cased and uncased bleu scores
    """
    tmp = tempfile.NamedTemporaryFile(delete=False)
    tmp_filename = tmp.name

    translate.translate_file(model, subtokenizer, bleu_source, output_file=tmp_filename,
                             print_all_translations=False)

    uncased_score = compute_bleu.bleu_wrapper(bleu_ref, tmp_filename, False)
    os.remove(tmp_filename)

    return uncased_score


def train_schedule(train_eval_iterations, single_iteration_train_steps, params,
                   bleu_source=None, bleu_ref=None,  bleu_threshold=None):
    """
    Train and evaluate model
    :param model: model to train
    :param train_eval_iterations: Number of times to repeat the train-eval iteration
    :param single_iteration_train_steps: Number of steps to train in one iteration
    :param bleu_source:File containing text to be translated for BLEU calculation.
    :param bleu_ref:File containing reference translations for BLEU calculation.
    :param bleu_threshold:minimum BLEU score before training is stopped.

    """
    print('Training schedule:')
    print('\t1.Train for %d iterations' % train_eval_iterations)
    print('\t2.Each iteration for %d steps.' % single_iteration_train_steps)
    print('\t3.Compute BLEU score.')
    '''if bleu_threshold is not None:
        print("Repeat above steps until the BLEU score reaches", bleu_threshold)
        train_eval_iterations = INF
    else:
        print("Repeat above steps %d times." % train_eval_iterations)'''

    # Loop training/evaluation/bleu cycles
    subtokenizer = tokenizer.Subtokenizer(vocab_file='vocab.ende.32768')
    dataset_train = dataset.TranslationDataset(dir_lang1='wmt32k-train.lang1',
                                               dir_lang2='wmt32k-train.lang2',
                                               subtokenizer=subtokenizer)
    global_step = 0
    best_bleu_score = 0
    net = transformer.Transformer(params=params, train=1)
    net.initialize(init=init.Xavier(), ctx=ctx, force_reinit=True)
    learning_rate = get_learning_rate(params.learning_rate, params.hidden_size,
                                      params.learning_rate_warmup_steps, global_step)
    optimizer = mx.optimizer.Adam(learning_rate=learning_rate, beta1=params.optimizer_adam_beta1,
                                  beta2=params.optimizer_adam_beta2, epsilon=params.optimizer_adam_epsilon)

    trainer = gluon.Trainer(net.collect_params(), optimizer=optimizer)
	bleu_score_file = open('blue_score_file', w+)
    for i in xrange(train_eval_iterations):
        gc.collect()
        print('Starting iteration', i + 1)
        print('Train:')
        for step in xrange(single_iteration_train_steps):
            tic = time()
            losses = 0
            mini_batch_train = dataset_train.get_mini_batch(batch_size=params.batch_size)
            input = gluon.utils.split_and_load(mini_batch_train['input'], ctx)
            targets = gluon.utils.split_and_load(mini_batch_train['targets'], ctx)
            global_step = 1 + global_step
            learning_rate = get_learning_rate(params.learning_rate, params.hidden_size,
                                              params.learning_rate_warmup_steps, global_step)
            with autograd.record():
                for j in xrange(num_gpu):
                    loss = metrics.padded_cross_entropy_loss(net(input[j], targets[j]), targets[j],
                                                             params.label_smoothing, params.vocab_size)
                    loss.backward()
                    losses = losses + loss
            trainer.set_learning_rate(learning_rate)
            trainer.step(params.batch_size)
            mx.ndarray.waitall()

            print("\t step %d: Loss: %.3f, Time:%.1f seconds" % (global_step, losses.mean().asscalar() / 4, time() - tic))

        print('Evaluate: ')
        uncased_score = translate_and_compute_bleu(net, subtokenizer, bleu_source, bleu_ref)
        print('\t uncased_score: %.3f' % uncased_score)
        print('\t best_bleu_score: %.3f' % best_bleu_score)

        if uncased_score > best_bleu_score:
            best_bleu_score = uncased_score
			blue_score_file.write("blue_score: %.3f" % best_bleu_score)
            net.save_parameters(filename='transformer.params')
	blue_score_file.close()


if __name__ == "__main__":
    train_eval_iterations = 10 # 10
    single_iteration_train_steps = 32000 # 2000
    params = model_params.TransformerBaseParams
    bleu_source = 'newstest2014.en'
    bleu_ref = 'newstest2014.de'
    bleu_threshold = 20
    train_schedule(train_eval_iterations, single_iteration_train_steps, params, bleu_source, bleu_ref, bleu_threshold)

