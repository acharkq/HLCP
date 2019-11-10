import tensorflow as tf
import sys
from collections import namedtuple
from models.customized_operations import masked_mean_pooling


def TextEncoder(name, *args, **kw):
    return getattr(sys.modules[__name__], name)(*args, **kw)


class EncoderOutput(
    namedtuple("EncoderOutput",
               ("outputs", "final_state", 'attention_values', 'attention_values_length', 'initial_state'))):
    pass


class Encoder(object):
    _sen_len = -1
    _dropout_keep_prob = 1
    _encoder_output = None

    def __init__(self, params, mode):
        self._sen_len = params['sen_len']
        self._dropout_keep_prob = params['dropout_keep_prob'] if mode == tf.estimator.ModeKeys.TRAIN else 1

    def __call__(self, input, input_length):
        raise NotImplementedError


class CNNEncoder(Encoder):
    _cnn_layer = None

    def __init__(self, params, mode):
        super(CNNEncoder, self).__init__(params, mode)
        self._cnn_layer = tf.layers.Conv2D(filters=params['num_filters'],
                                           kernel_size=[params['kernel_size'], params['embedding_size']],
                                           padding='valid',
                                           activation=tf.nn.relu,
                                           name='cnn_layer', )

    def __call__(self, input, input_length):
        kernel_size = self._cnn_layer.kernel_size[0]

        input = tf.expand_dims(input, -1)
        conv_output = self._cnn_layer.apply(input)
        conv_output = tf.nn.dropout(conv_output, self._dropout_keep_prob)
        conv_output = tf.squeeze(conv_output, axis=2)
        max_pooled = tf.layers.max_pooling1d(conv_output, self._sen_len - kernel_size + 1,
                                             strides=1)
        max_pooled = tf.squeeze(max_pooled, axis=1, name='max_pooled')
        # calculate the output length of convolution layer
        conv_length = input_length - kernel_size + 1

        # manual masked mean pooling
        first_attention = masked_mean_pooling(conv_output, conv_length,
                                              self._sen_len - kernel_size + 1)
        return EncoderOutput(outputs=conv_output, final_state=first_attention, attention_values=conv_output,
                             attention_values_length=conv_length, initial_state=max_pooled)


class LSTMEncoder(Encoder):
    _cell = None

    def __init__(self, params, mode):
        super(LSTMEncoder, self).__init__(params, mode)
        cell = tf.nn.rnn_cell.BasicLSTMCell(num_units=params['num_units'])
        cell = tf.nn.rnn_cell.DropoutWrapper(cell, output_keep_prob=self._dropout_keep_prob)
        self._cell = tf.nn.rnn_cell.MultiRNNCell([cell] * params['num_layers'])

    def __call__(self, inputs, input_lengths):
        cell_initial_state = self._cell.zero_state(tf.shape(inputs)[0], tf.float32)
        rnn_outputs, _ = tf.nn.dynamic_rnn(self._cell, inputs, initial_state=cell_initial_state,
                                           sequence_length=input_lengths, parallel_iterations=128)
        # rnn_outputs shape = [batch_size, num_steps, unit_num]
        first_attention = masked_mean_pooling(rnn_outputs, input_lengths, self._sen_len)
        max_pooled = tf.squeeze(tf.layers.max_pooling1d(rnn_outputs, self._sen_len, 1, name='pool'),
                                axis=1)
        return EncoderOutput(outputs=rnn_outputs, final_state=first_attention, attention_values=rnn_outputs,
                             attention_values_length=input_lengths, initial_state=max_pooled)


class BiLSTMEncoder(Encoder):

    def __init__(self, params, mode):
        super(BiLSTMEncoder, self).__init__(params, mode)
        self._cell_fw = tf.nn.rnn_cell.BasicLSTMCell(num_units=params['num_units'])
        self._cell_fw = tf.nn.rnn_cell.DropoutWrapper(self._cell_fw, output_keep_prob=self._dropout_keep_prob)
        self._cell_bw = tf.nn.rnn_cell.BasicLSTMCell(num_units=params['num_units'])
        self._cell_bw = tf.nn.rnn_cell.DropoutWrapper(self._cell_bw, output_keep_prob=self._dropout_keep_prob)

    def __call__(self, inputs, input_lengths):
        cell_fw_initial_state = self._cell_fw.zero_state(tf.shape(inputs)[0], tf.float32)
        cell_bw_initial_state = self._cell_bw.zero_state(tf.shape(inputs)[0], tf.float32)
        outputs, output_states = tf.nn.bidirectional_dynamic_rnn(cell_fw=self._cell_fw, cell_bw=self._cell_bw,
                                                                 inputs=inputs, sequence_length=input_lengths,
                                                                 initial_state_fw=cell_fw_initial_state,
                                                                 initial_state_bw=cell_bw_initial_state,
                                                                 parallel_iterations=128)
        outputs = tf.concat(outputs, axis=-1)
        max_pooled = tf.squeeze(tf.layers.max_pooling1d(outputs, self._sen_len, 1, name='pool'),
                                axis=1)
        # rnn_outputs shape = [batch_size, num_steps, unit_num]
        first_attention = masked_mean_pooling(outputs, input_lengths, self._sen_len)
        return EncoderOutput(outputs=outputs, final_state=first_attention, attention_values=outputs,
                             attention_values_length=input_lengths, initial_state=max_pooled)
