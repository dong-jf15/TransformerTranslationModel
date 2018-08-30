from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from mxnet.gluon import nn
from mxnet import nd
import mxnet as mx

import model.attention_layer as attention_layer
import model.beam_search as beam_search
import model.embedding_layer as embedding_layer
import model.fnn_layer as fnn_layer
import model.model_utils as model_utils
import model.model_params as model_params
from utils.tokenizer import EOS_ID

_NEG_INF = -1e9


class Transformer(nn.Block):
    """Transformer model for sequence to sequence data.
    The Transformer model consists of an encoder and decoder. The input is an int
    sequence (or a batch of sequences). The encoder produces a continous
    representation, and the decoder uses the encoder output to generate
    probabilities for the output sequence.
    """
    def __init__(self, params, train, **kwargs):
        super(Transformer, self).__init__(**kwargs)
        self.train = train
        self.param = params

        with self.name_scope():
            self.embedding_softmax_layer = embedding_layer.EmbeddingSharedWeights(params.vocab_size, params.hidden_size)
            self.encoder_stack = EncoderStack(params, train)
            self.decoder_stack = DecoderStack(params, train)
            self.dropout_input = nn.Dropout(1-self.param.layer_postprocess_dropout)
            self.dropout_output = nn.Dropout(1-self.param.layer_postprocess_dropout)

    def forward(self, inputs, targets=None):
        """Calculate target logits or inferred target sequences.

        Args:
         inputs: int tensor with shape [batch_size, input_length].
         targets: None or int tensor with shape [batch_size, target_length].

        Returns:
          If targets is defined, then return logits for each word in the target
          sequence. float tensor with shape [batch_size, target_length, vocab_size]
          If target is none, then generate output sequence one token at a time.
          returns a dictionary {
            output: [batch_size, decoded length]
            score: [batch_size, float]}
        """
        attention_bias = model_utils.get_padding_bias(inputs)
        encoder_outputs = self.encode(inputs, attention_bias)
        if targets is None:
            return self.predict(encoder_outputs, attention_bias)
        else:
            logits = self.decode(targets, encoder_outputs, attention_bias)
            return logits

    def encode(self, inputs, attention_bias):
        """Generate continuous representation for inputs.

        Args:
            inputs: int tensor with shape [batch_size, input_length].
            attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

        Returns:
            float tensor with shape [batch_size, input_length, hidden_size]
        """
        embedded_inputs = self.embedding_softmax_layer(inputs)
        inputs_padding = model_utils.get_padding(inputs)

        length = embedded_inputs.shape[1]
        pos_encoding = model_utils.get_position_encoding(length, self.param.hidden_size, inputs.context)
        encoder_inputs = embedded_inputs + pos_encoding

        if self.train:
            encoder_inputs = self.dropout_input(encoder_inputs)

        return self.encoder_stack(encoder_inputs, attention_bias, inputs_padding)

    def decode(self, targets, encoder_outputs, attention_bias):
        """Generate logits for each value in the target sequence.

            Args:
              targets: target values for the output sequence.
                int tensor with shape [batch_size, target_length]
              encoder_outputs: continuous representation of input sequence.
                float tensor with shape [batch_size, input_length, hidden_size]
              attention_bias: float tensor with shape [batch_size, 1, 1, input_length]

            Returns:
              float32 tensor with shape [batch_size, target_length, vocab_size]
        """
        decoder_inputs = self.embedding_softmax_layer(targets)
        decoder_inputs = nd.expand_dims(decoder_inputs, axis=0)
        decoder_inputs = nd.pad(data=decoder_inputs, mode="constant", constant_value=0, pad_width=(0,0,0,0,1,0,0,0))
        decoder_inputs = nd.reshape(data=decoder_inputs, shape=decoder_inputs.shape[1:])[:, :-1, :]

        length = decoder_inputs.shape[1]
        decoder_inputs = decoder_inputs + model_utils.get_position_encoding(length, self.param.hidden_size,
                                                                            targets.context)
        if self.train:
            decoder_inputs = self.dropout_output(decoder_inputs)

        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(length, targets.context)
        outputs = self.decoder_stack(decoder_inputs, encoder_outputs, decoder_self_attention_bias, attention_bias)
        logits = self.embedding_softmax_layer.linear(outputs)
        return logits

    def _get_symbols_to_logits_fn(self, max_decode_length):
        """Returns a decoding function that calculates logits of the next tokens."""
        timing_signal = model_utils.get_position_encoding(
            max_decode_length + 1, self.param.hidden_size, mx.gpu())
        decoder_self_attention_bias = model_utils.get_decoder_self_attention_bias(
            max_decode_length, mx.gpu())

        def symbols_to_logits_fn(ids, i, cache):
            decoder_input = ids[:, -1:]
            # decoder的输入为Current decoded sequences 的最后一个

            decoder_input = self.embedding_softmax_layer(decoder_input)
            decoder_input = decoder_input + timing_signal[i:i + 1]

            self_attention_bias = decoder_self_attention_bias[:, :, i:i + 1, :i + 1]
            decoder_outputs = self.decoder_stack(
                decoder_input, cache.get("encoder_outputs"), self_attention_bias,
                cache.get("encoder_decoder_attention_bias"), cache)
            logits = self.embedding_softmax_layer.linear(decoder_outputs)

            logits = nd.squeeze(logits, axis=1)
            return logits, cache
        return symbols_to_logits_fn

    def predict(self, encoder_outputs, encoder_decoder_attention_bias):
        """Return predicted sequence."""
        batch_size = encoder_outputs.shape[0]
        input_length = encoder_outputs.shape[1]
        max_decode_length = input_length + self.param.extra_decode_length

        symbols_to_logits_fn = self._get_symbols_to_logits_fn(max_decode_length)

        initial_ids = nd.zeros(shape=batch_size, ctx=mx.cpu())
        # Create cache storing decoder attention values for each layer.
        '''cache = {
            "layer_%d" % layer: {
                "k": nd.zeros(shape=(batch_size, 0, self.param.hidden_size), ctx=ctx),
                "v": nd.zeros(shape=(batch_size, 0, self.param.hidden_size), ctx=ctx),
            } for layer in range(self.param.num_hidden_layers)}'''
        cache = {}
        for layer in range(self.param.num_hidden_layers):
            cache["layer_%d" % layer] = {
                "k": nd.zeros(shape=(batch_size, 1, self.param.hidden_size), ctx=mx.cpu()),
                "v": nd.zeros(shape=(batch_size, 1, self.param.hidden_size), ctx=mx.cpu()),
                "init": 1}

        cache["encoder_outputs"] = encoder_outputs.as_in_context(mx.cpu())
        cache["encoder_decoder_attention_bias"] = encoder_decoder_attention_bias.as_in_context(mx.cpu())

        decoded_ids, scores = beam_search.sequence_beam_search(
            symbols_to_logits_fn=symbols_to_logits_fn,
            initial_ids=initial_ids,
            initial_cache=cache,
            vocab_size=self.param.vocab_size,
            beam_size=self.param.beam_size,
            alpha=self.param.alpha,
            max_decode_length=max_decode_length,
            eos_id=EOS_ID)

        top_decoded_ids = decoded_ids[:, 0, 1:]
        top_scores = scores[:, 0]

        return {"outputs": top_decoded_ids, "scores": top_scores}


class PrePostProcessingWrapper(nn.Block):
    """Actually ADD & NORM"""
    def __init__(self, layer, params, train, **kwargs):
        super(PrePostProcessingWrapper, self).__init__(**kwargs)
        self.postprocess_dropout = params.layer_postprocess_dropout
        self.train = train

        with self.name_scope():
            self.layer = layer
            self.layer_norm = nn.LayerNorm(epsilon=1e-6)
            self.dropout = nn.Dropout(1 - self.postprocess_dropout)

    def forward(self, x, *args, **kwargs):
        y = self.layer_norm(x)
        y = self.layer(y, *args, **kwargs)

        if self.train:
            y = self.dropout(y)
        return x + y


class EncoderStack(nn.Block):
    """Transformer encoder stack.

    The encoder stack is made up of N identical layers. Each layer is composed
    of the sublayers:
      1. Self-attention layer
      2. Feedforward network (which is 2 fully-connected layers)
    """

    def __init__(self, params, train, **kwargs):
        super(EncoderStack, self).__init__(**kwargs)
        self.param = params
        with self.name_scope():
            self.layer = nn.Sequential()
            with self.layer.name_scope():
                for i in range(params.num_hidden_layers):
                    self_attention_layer = attention_layer.SelfAttention(
                        params.hidden_size, params.num_heads, params.attention_dropout, train)
                    feed_forward_network = fnn_layer.FeedForwardNetwork(
                        params.hidden_size, params.filter_size, params.relu_dropout, train)

                    self.layer.add(PrePostProcessingWrapper(self_attention_layer, params, train))
                    self.layer.add(PrePostProcessingWrapper(feed_forward_network, params, train))

            self.output_normalization = nn.LayerNorm(axis=-1, epsilon=1e-6)

    def forward(self, encoder_inputs, attention_bias, inputs_padding):
        for i in range(0, 2*self.param.num_hidden_layers, 2):
            encoder_inputs = self.layer[i](encoder_inputs, attention_bias)
            encoder_inputs = self.layer[i+1](encoder_inputs, inputs_padding)

        return self.output_normalization(encoder_inputs)


class DecoderStack(nn.Block):
    """Transformer decoder stack.

    Like the encoder stack, the decoder stack is made up of N identical layers.
    Each layer is composed of the sublayers:
      1. Self-attention layer
      2. Multi-headed attention layer combining encoder outputs with results from
         the previous self-attention layer.
      3. Feedforward network (2 fully-connected layers)
    """
    def __init__(self, params, train, **kwargs):
        super(DecoderStack, self).__init__(**kwargs)
        self.param = params
        with self.name_scope():
            self.layer = nn.Sequential()
            with self.layer.name_scope():
                for i in range(params.num_hidden_layers):
                    self_attention_layer = attention_layer.SelfAttention(
                        params.hidden_size, params.num_heads, params.attention_dropout, train)
                    enc_dec_attention_layer = attention_layer.Attention(
                        params.hidden_size, params.num_heads, params.attention_dropout, train)
                    feed_forward_network = fnn_layer.FeedForwardNetwork(
                        params.hidden_size, params.filter_size, params.relu_dropout, train)

                    self.layer.add(PrePostProcessingWrapper(self_attention_layer, params, train),
                                   PrePostProcessingWrapper(enc_dec_attention_layer, params, train),
                                   PrePostProcessingWrapper(feed_forward_network, params, train))
            self.output_normalization = nn.LayerNorm(axis=-1, epsilon=1e-6)

    def forward(self, decoder_inputs, encoder_outputs, decoder_self_attention_bias,
           attention_bias, cache=None):
        for i in range(0, 3*self.param.num_hidden_layers, 3):
            layer_name = "layer_%d" % (i / 3)
            layer_cache = cache[layer_name] if cache is not None else None
            decoder_inputs = self.layer[i](decoder_inputs, decoder_self_attention_bias, layer_cache)
            decoder_inputs = self.layer[i+1](decoder_inputs, encoder_outputs, attention_bias)
            decoder_inputs = self.layer[i+2](decoder_inputs)

        return self.output_normalization(decoder_inputs)


if __name__ == "__main__":

    param = model_params.TransformerBaseParams()
    net = Transformer(params=param, train=1)
    net.initialize(ctx=mx.gpu())
    print(net.collect_params())
    input = nd.array([[4, 2, 10, 12, 1],[2, 3, 5, 7, 1]], ctx=mx.gpu())
    targets = nd.array([[100, 200, 300, 400, 1], [500, 600, 700, 800, 1]], ctx=mx.gpu())
    output = net(input)
    print(output)
