"""Implement of beam_search"""
from mxnet import nd
import numpy as np
import mxnet as mx
from six.moves import xrange

INF = 1. * 1e7
max_iterations = 200
ctx = mx.cpu()


class _StateKeys(object):
    """Keys to dictionary storing the state of the beam search loop."""

    # Variable storing the loop index.
    CUR_INDEX = "CUR_INDEX"

    # Top sequences that are alive for each batch item. Alive sequences are ones
    # that have not generated an EOS token. Sequences that reach EOS are marked as
    # finished and moved to the FINISHED_SEQ tensor.
    # Has shape [batch_size, beam_size, CUR_INDEX + 1]
    ALIVE_SEQ = "ALIVE_SEQ"
    # Log probabilities of each alive sequence. Shape [batch_size, beam_size]
    ALIVE_LOG_PROBS = "ALIVE_LOG_PROBS"
    # Dictionary of cached values for each alive sequence. The cache stores
    # the encoder output, attention bias, and the decoder attention output from
    # the previous iteration.
    ALIVE_CACHE = "ALIVE_CACHE"
    # Top finished sequences for each batch item.
    # Has shape [batch_size, beam_size, CUR_INDEX + 1]. Sequences that are
    # shorter than CUR_INDEX + 1 are padded with 0s.
    FINISHED_SEQ = "FINISHED_SEQ"
    # Scores for each finished sequence. Score = log probability / length norm
    # Shape [batch_size, beam_size]
    FINISHED_SCORES = "FINISHED_SCORES"
    # Flags indicating which sequences in the finished sequences are finished.
    # At the beginning, all of the sequences in FINISHED_SEQ are filler values.
    # True -> finished sequence, False -> filler. Shape [batch_size, beam_size]
    FINISHED_FLAGS = "FINISHED_FLAGS"


class SequenceBeamSearch(object):
    """Implementation of beam search loop."""
    def __init__(self, symbols_to_logits_fn, vocab_size, batch_size,
                 beam_size, alpha, max_decode_length, eos_id):
        self.symbols_to_logits_fn = symbols_to_logits_fn
        self.vocab_size = vocab_size
        self.batch_size = batch_size
        self.beam_size = beam_size
        self.alpha = alpha
        self.max_decode_length = max_decode_length
        self.eos_id = eos_id

    def search(self, initial_ids, initial_cache):
        """Beam search for sequences with highest scores."""
        state = self._create_initial_state(initial_ids, initial_cache)

        while self._continue_search(state):
            state = self._search_step(state)

        finished_state = state
        alive_seq = finished_state[_StateKeys.ALIVE_SEQ]
        alive_log_probs = finished_state[_StateKeys.ALIVE_LOG_PROBS]
        finished_seq = finished_state[_StateKeys.FINISHED_SEQ]
        finished_scores = finished_state[_StateKeys.FINISHED_SCORES]
        finished_flags = finished_state[_StateKeys.FINISHED_FLAGS]

        finished_seq = nd.where(nd.array(np.any(finished_flags.asnumpy(), axis=1), ctx=ctx), finished_seq, alive_seq)
        finished_scores = nd.where(nd.array(np.any(finished_flags.asnumpy(), axis=1), ctx=ctx),
                                   finished_scores, alive_log_probs)
        return finished_seq, finished_scores

    def _create_initial_state(self, initial_ids, initial_cache):
        """Return initial state dictionary and its shape invariants.

        Args:
          initial_ids: initial ids to pass into the symbols_to_logits_fn.
            int tensor with shape [batch_size, 1]
          initial_cache: dictionary storing values to be passed into the
            symbols_to_logits_fn.

        Returns:
            state and shape invariant dictionaries with keys from _StateKeys
        """
        cur_index = 0

        alive_seq = _expand_to_beam_size(initial_ids, self.beam_size)
        alive_seq = nd.expand_dims(alive_seq, axis=2)

        initial_log_probs = nd.array([[0.] + [-float("inf")] * (self.beam_size - 1)], ctx=ctx)
        alive_log_probs = nd.tile(initial_log_probs, (self.batch_size, 1))

        alive_cache = map_structure(lambda t: _expand_to_beam_size(t, self.beam_size), initial_cache)
        finished_seq = nd.zeros(alive_seq.shape, ctx=ctx)
        finished_scores = nd.ones((self.batch_size, self.beam_size), ctx=ctx) * -INF
        finished_flags = nd.zeros((self.batch_size, self.beam_size), ctx=ctx)

        state = {
            _StateKeys.CUR_INDEX: cur_index,
            _StateKeys.ALIVE_SEQ: alive_seq,
            _StateKeys.ALIVE_LOG_PROBS: alive_log_probs,
            _StateKeys.ALIVE_CACHE: alive_cache,
            _StateKeys.FINISHED_SEQ: finished_seq,
            _StateKeys.FINISHED_SCORES: finished_scores,
            _StateKeys.FINISHED_FLAGS: finished_flags
        }
        return state

    def _continue_search(self, state):
        """Return whether to continue the search loop.

        The loops should terminate when
          1) when decode length has been reached, or
          2) when the worst score in the finished sequences is better than the best
             score in the alive sequences (i.e. the finished sequences are provably
             unchanging)

        Args:
          state: A dictionary with the current loop state.

        Returns:
          Bool tensor with value True if loop should continue, False if loop should
          terminate.
        """
        i = state[_StateKeys.CUR_INDEX]
        alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS]
        finished_scores = state[_StateKeys.FINISHED_SCORES]
        finished_flags = state[_StateKeys.FINISHED_FLAGS]

        not_at_max_decode_length = i < self.max_decode_length

        max_length_norm = _length_normalization(self.alpha, self.max_decode_length)

        best_alive_scores = alive_log_probs[:, 0] / max_length_norm

        finished_scores = finished_flags * finished_scores
        lowest_finished_scores = nd.min(finished_scores, axis=1)

        finished_batches = nd.array(np.any(finished_flags.asnumpy(), axis=1), ctx=ctx)
        lowest_finished_scores = lowest_finished_scores + (1. - finished_batches) * -INF

        worst_finished_score_better_than_best_alive_score = np.all(lowest_finished_scores.asnumpy() >
                                                                   best_alive_scores.asnumpy())
        return not_at_max_decode_length and not worst_finished_score_better_than_best_alive_score

    def _search_step(self, state):
        """Beam search loop body.

        Grow alive sequences by a single ID. Sequences that have reached the EOS
        token are marked as finished. The alive and finished sequences with the
        highest log probabilities and scores are returned.

        A sequence's finished score is calculating by dividing the log probability
        by the length normalization factor. Without length normalization, the
        search is more likely to return shorter sequences.

        Args:
          state: A dictionary with the current loop state.

        Returns:
          new state dictionary.
        """
        new_seq, new_log_probs, new_cache = self._grow_alive_seq(state)
        alive_state = self._get_new_alive_state(new_seq, new_log_probs, new_cache)

        finished_state = self._get_new_finished_state(state, new_seq, new_log_probs)

        new_state = {_StateKeys.CUR_INDEX: state[_StateKeys.CUR_INDEX] + 1}
        new_state.update(alive_state)
        new_state.update(finished_state)

        return new_state

    def _get_new_finished_state(self, state, new_seq, new_log_probs):
        """Combine new and old finished sequences, and gather the top k sequences.

        Args:
          state: A dictionary with the current loop state.
          new_seq: New sequences generated by growing the current alive sequences
            int32 tensor with shape [batch_size, beam_size, i + 1]
          new_log_probs: Log probabilities of new sequences
            float32 tensor with shape [batch_size, beam_size]

        Returns:
          Dictionary with finished keys from _StateKeys:
            {Top beam_size finished sequences based on score,
             Scores of finished sequences,
             Finished flags of finished sequences}
        """
        i = state[_StateKeys.CUR_INDEX]
        finished_seq = state[_StateKeys.FINISHED_SEQ]
        finished_scores = state[_StateKeys.FINISHED_SCORES]
        finished_flags = state[_StateKeys.FINISHED_FLAGS]

        finished_seq = nd.concat(finished_seq, nd.zeros(shape=(self.batch_size, self.beam_size, 1), ctx=ctx), dim=2)
        length_norm = _length_normalization(self.alpha, i + 1)
        new_scores = new_log_probs / length_norm

        new_finished_flags = nd.equal(new_seq[:, :, -1], self.eos_id)
        new_scores = new_scores + (1. - new_finished_flags) * -INF

        # combine sequences, scores, and flags
        finished_seq = nd.concat(finished_seq, new_seq, dim=1)
        finished_scores = nd.concat(finished_scores, new_scores, dim=1)
        finished_flags = nd.concat(finished_flags, new_finished_flags, dim=1)

        top_finished_seq, top_finished_scores, top_finished_flags = _gather_topk_beams(
            [finished_seq, finished_scores, finished_flags], finished_scores, self.batch_size, self.beam_size)

        return {
            _StateKeys.FINISHED_SEQ: top_finished_seq,
            _StateKeys.FINISHED_SCORES: top_finished_scores,
            _StateKeys.FINISHED_FLAGS: top_finished_flags
        }

    def _get_new_alive_state(self, new_seq, new_log_probs, new_cache):
        """Gather the top k sequences that are still alive.

        Args:
          new_seq: New sequences generated by growing the current alive sequences
            int32 tensor with shape [batch_size, 2 * beam_size, cur_index + 1]
          new_log_probs: Log probabilities of new sequences
            float32 tensor with shape [batch_size, beam_size]
          new_cache: Dict of cached values for each sequence.

        Returns:
          Dictionary with alive keys from _StateKeys:
            {Top beam_size sequences that are still alive (don't end with eos_id)
             Log probabilities of top alive sequences
             Dict cache storing decoder states for top alive sequences}
        """
        new_finished_flags = nd.equal(new_seq[:, :, -1], self.eos_id)
        new_log_probs = new_log_probs + new_finished_flags * -INF
        top_alive_seq, top_alive_log_probs = _gather_topk_beams(
            [new_seq, new_log_probs], new_log_probs, self.batch_size, self.beam_size)
        top_alive_cache = _gather_topk_beams([], new_log_probs, self.batch_size, self.beam_size, cache=new_cache)

        return {
            _StateKeys.ALIVE_SEQ: top_alive_seq,
            _StateKeys.ALIVE_LOG_PROBS: top_alive_log_probs,
            _StateKeys.ALIVE_CACHE: top_alive_cache
        }

    def _grow_alive_seq(self, state):
        """Grow alive sequences by one token, and collect top 2*beam_size sequences.

        2*beam_size sequences are collected because some sequences may have reached
        the EOS token. 2*beam_size ensures that at least beam_size sequences are
        still alive.

        Args:
          state: A dictionary with the current loop state.
        Returns:
          Tuple of
          (Top 2*beam_size sequences [batch_size, 2 * beam_size, cur_index + 1],
           Scores of returned sequences [batch_size, 2 * beam_size],
           New alive cache, for each of the 2 * beam_size sequences)
        """
        i = state[_StateKeys.CUR_INDEX]
        alive_seq = state[_StateKeys.ALIVE_SEQ]
        alive_log_probs = state[_StateKeys.ALIVE_LOG_PROBS]
        alive_cache = state[_StateKeys.ALIVE_CACHE]

        beams_to_keep = 2 * self.beam_size

        flat_ids = _flatten_beam_dim(alive_seq)

        # put the input on the gpu device
        flat_cache = map_structure(_flatten_beam_dim, alive_cache, context=mx.gpu())
        flat_ids = flat_ids.as_in_context(mx.gpu())

        # use the model to predict
        flat_logits, flat_cache = self.symbols_to_logits_fn(flat_ids, i, flat_cache)

        # put the output on the cpu device
        flat_logits = flat_logits.as_in_context(ctx)
        flat_cache = map_structure(lambda t: t, flat_cache, context=ctx)

        logits = _unflatten_beam_dim(flat_logits, self.batch_size, self.beam_size)

        new_cache = map_structure(lambda t: _unflatten_beam_dim(t, self.batch_size, self.beam_size), flat_cache)

        candidate_log_probs = _log_prob_from_logits(logits)

        log_probs = candidate_log_probs + nd.expand_dims(alive_log_probs, axis=2)

        flat_log_probs = nd.reshape(log_probs, (-1, self.beam_size * self.vocab_size))

        topk_log_probs, topk_indices = nd.topk(flat_log_probs, k=beams_to_keep, ret_typ='both')

        topk_beam_indices = nd.array(topk_indices.asnumpy() // self.vocab_size, ctx=ctx)
        topk_seq = _gather_beams([alive_seq], topk_beam_indices, self.batch_size, beams_to_keep)
        new_cache = _gather_beams([], topk_beam_indices, self.batch_size, beams_to_keep, cache=new_cache)

        topk_ids = topk_indices % self.vocab_size
        topk_ids = nd.expand_dims(topk_ids, axis=2)
        topk_seq = nd.concat(topk_seq[0], topk_ids, dim=2)

        return topk_seq, topk_log_probs, new_cache


def _gather_topk_beams(list, score_or_log_probs, batch_size, beam_size, cache=None):
    """Gather top beams from nested structure."""
    score_or_log_probs = nd.array(score_or_log_probs, ctx=ctx)
    topk_indexes = nd.topk(score_or_log_probs, k=beam_size)

    return _gather_beams(list, topk_indexes, batch_size, beam_size, cache=cache)


def _log_prob_from_logits(logits):
    tmp = nd.exp(logits)
    tmp = nd.sum(tmp, axis=2, keepdims=True)
    tmp = nd.log(tmp)
    return logits - tmp


def _gather_beams(list, beam_indices, batch_size, new_beam_size, cache=None):
    """Gather beams from nested structure of tensors.

    Each tensor in nested represents a batch of beams, where beam refers to a
    single search state (beam search involves searching through multiple states
    in parallel).

    This function is used to gather the top beams, specified by
    beam_indices, from the nested tensors.

    Args:
      nested: Nested structure (tensor, list, tuple or dict) containing tensors
        with shape [batch_size, beam_size, ...].
      beam_indices: int32 tensor with shape [batch_size, new_beam_size]. Each
       value in beam_indices must be between [0, beam_size), and are not
       necessarily unique.
      batch_size: int size of batch
      new_beam_size: int number of beams to be pulled from the nested tensors.

    Returns:
      Nested structure containing tensors with shape
        [batch_size, new_beam_size, ...]
    """
    batch_pos = np.arange(0, batch_size * new_beam_size)
    batch_pos = nd.array(batch_pos, ctx=ctx, dtype='int32') / new_beam_size
    batch_pos = nd.reshape(batch_pos, (batch_size, new_beam_size))
    beam_indices = nd.cast(beam_indices, dtype='int32')

    coordinates = nd.stack(batch_pos, beam_indices, axis=2)
    m = coordinates.shape[0]
    n = coordinates.shape[1]
    coordinates_tmp = nd.zeros(shape=(m, 2, n), ctx=ctx)
    for i in xrange(m):
        coordinates_tmp[i] = coordinates[i].T

    coordinates_new = nd.ones(shape=(2, m, n), ctx=ctx)
    for i in xrange(m):
        coordinates_new[0][i] = coordinates_tmp[i][0]
        coordinates_new[1][i] = coordinates_tmp[i][1]

    if cache is None:
        for i in xrange(len(list)):
            list[i] = nd.gather_nd(list[i], coordinates_new)
        return list
    else:
        cache = map_structure(lambda t: nd.gather_nd(t, coordinates_new), cache)
        return cache


def _unflatten_beam_dim(array, batch_size, beam_size):
    """Reshapes first dimension back to [batch_size, beam_size].

    Args:
      tensor: Tensor to reshape of shape [batch_size*beam_size, ...]
      batch_size: Tensor, original batch size.
      beam_size: int, original beam size.

    Returns:
      Reshaped tensor of shape [batch_size, beam_size, ...]
    """
    shape = list(array.shape)
    shape[1] = -1
    shape = tuple([batch_size, beam_size] + shape[1:])
    return nd.reshape(array, shape=shape)


def _expand_to_beam_size(array, beam_size):
    """Tiles a given tensor by beam_size.

    Args:
      tensor: tensor to tile [batch_size, ...]
      beam_size: How much to tile the tensor by.

    Returns:
      Tiled tensor [batch_size, beam_size, ...]
    """
    array = nd.expand_dims(array, axis=1)
    tile_dims = [1] * array.ndim
    tile_dims[1] = beam_size

    return nd.tile(array, tile_dims)


def _length_normalization(alpha, length):
    return nd.power(((nd.array([5.], ctx=ctx) + length) / 6.0), alpha)


def _flatten_beam_dim(array):
    """Reshapes first two dimensions in to single dimension.

    Args:
      tensor: Tensor to reshape of shape [A, B, ...]

    Returns:
      Reshaped tensor of shape [A*B, ...]
    """
    shape = list(array.shape)
    shape[1] = shape[1] * shape[0]
    shape[2] = -1
    shape = tuple(shape[1:])
    return nd.reshape(array, shape)


def map_structure(function, cache, context=None):
    if context is None:
        for layer in xrange(6):
            str = "layer_%d" % layer
            cache[str]['k'] = function(cache[str]['k'])
            cache[str]['v'] = function(cache[str]['v'])
        cache["encoder_outputs"] = function(cache["encoder_outputs"])
        cache["encoder_decoder_attention_bias"] = function(cache["encoder_decoder_attention_bias"])
        return cache
    else:
        for layer in xrange(6):
            str = "layer_%d" % layer
            cache[str]['k'] = function(cache[str]['k']).as_in_context(context)
            cache[str]['v'] = function(cache[str]['v']).as_in_context(context)

        cache["encoder_outputs"] = function(cache["encoder_outputs"]).as_in_context(context)
        cache["encoder_decoder_attention_bias"] = \
            function(cache["encoder_decoder_attention_bias"]).as_in_context(context)
        return cache


def sequence_beam_search(symbols_to_logits_fn, initial_ids, initial_cache, vocab_size, beam_size,
                         alpha, max_decode_length, eos_id):
    """Search for sequence of subtoken ids with the largest probability.

    Args:
      symbols_to_logits_fn: A function that takes in ids, index, and cache as
        arguments. The passed in arguments will have shape:
          ids -> [batch_size * beam_size, index]
          index -> [] (scalar)
          cache -> nested dictionary of tensors [batch_size * beam_size, ...]
        The function must return logits and new cache.
          logits -> [batch * beam_size, vocab_size]
          new cache -> same shape/structure as inputted cache
      initial_ids: Starting ids for each batch item.
        int32 tensor with shape [batch_size]
      initial_cache: dict containing starting decoder variables information
      vocab_size: int size of tokens
      beam_size: int number of beams
      alpha: float defining the strength of length normalization
      max_decode_length: maximum length to decoded sequence
      eos_id: int id of eos token, used to determine when a sequence has finished

    Returns:
      Top decoded sequences [batch_size, beam_size, max_decode_length]
      sequence scores [batch_size, beam_size]
    """
    batch_size = initial_ids.shape[0]
    sbs = SequenceBeamSearch(symbols_to_logits_fn, vocab_size, batch_size, beam_size, alpha, max_decode_length, eos_id)
    output = sbs.search(initial_ids, initial_cache)

    return output
