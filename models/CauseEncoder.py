import tensorflow as tf
import sys
from tensorflow.contrib import rnn
from tensorflow.python.ops import gen_array_ops


def NameEncoder(name, *args, **kw):
    return getattr(sys.modules[__name__], name)(*args, **kw)


class CauseEncoder(object):
    ''''updated on 2018-7-28 for cause buchang'''

    def __init__(self, word_embeddings, params):
        self._word_embeddings = word_embeddings
        self._params = params
        self._embedding_size = int(self._word_embeddings._dimension_size)
        # self._lstm_cell = rnn.BasicLSTMCell(num_units=self._embedding_size, reuse=tf.AUTO_REUSE)
        self._lstm_cell = rnn.LayerNormBasicLSTMCell(num_units=self._embedding_size, reuse=tf.AUTO_REUSE)
        self._cause_word_table = tf.constant(params['cause_word_table'], name='cause_word_table')
        self._cause_word_table_length = tf.constant(params['cause_word_table_length'], name='cause_word_table_length')

    def __call__(self, inputs):
        return self.apply(inputs)

    def apply(self, inputs):
        '''
        inputs shape = [batch_size, max_num_causes] the word id of input cause
        sequence_length = [batch_size,]
        return shape = [batch_size, max_cause_num, embedding_size]
        '''
        max_num_causes = int(inputs.shape[1])
        max_word_num = int(self._cause_word_table.shape[1])
        inputs_word_ids = gen_array_ops.gather_v2(self._cause_word_table, inputs, axis=0)
        # shape = [batch_size, max_num_causes, max_words_length]
        embedded_inputs = self._word_embeddings(inputs_word_ids)
        # shape = [batch_size, max_num_causes, max_words_length, embedding_size], and it will be flattend
        embedded_inputs = tf.reshape(embedded_inputs, [-1, max_word_num, self._embedding_size])
        inputs_word_length = gen_array_ops.gather_v2(self._cause_word_table_length, inputs, axis=0)
        # shape = [batch_size, max_num_causes]
        inputs_word_length = tf.reshape(inputs_word_length, [-1])
        lstm_zero_state = self._lstm_cell.zero_state(tf.shape(inputs_word_length)[0], tf.float32)
        outputs, state = tf.nn.dynamic_rnn(self._lstm_cell, embedded_inputs, inputs_word_length, lstm_zero_state,
                                           scope='lstm_cause_encoder')
        # state = [c, h] shape = [batch_size * max_num_causes, embedding_size]
        state = tf.reshape(state[0], [-1, max_num_causes, self._embedding_size])
        return state


class CauseEncoder_v2(CauseEncoder):
    _cause_embeddings = None

    def __init__(self, word_embeddings, params):
        super(CauseEncoder_v2, self).__init__(word_embeddings, params)
        self._cause_embeddings = tf.nn.l2_normalize(
            tf.random_uniform([params['num_causes'], self._embedding_size], -1.0, 1.0), axis=1)
        self._cause_embeddings = tf.get_variable(name='cause_embeddings', initializer=self._cause_embeddings,
                                                 dtype=tf.float32)

    def apply(self, inputs):
        lstm_state = super(CauseEncoder_v2, self).apply(inputs)
        embedding_state = gen_array_ops.gather_v2(self._cause_embeddings, inputs, axis=0)
        state = tf.concat([lstm_state, embedding_state], axis=2)
        return state


class RandomEncoder(object):
    _cause_embeddings = None

    def __init__(self, params, **kwargs):
        self._embedding_size = params['embedding_size']
        self._cause_embeddings = tf.nn.l2_normalize(
            tf.random_uniform([params['num_causes'], self._embedding_size], -1.0, 1.0), axis=1)
        self._cause_embeddings = tf.get_variable(name='cause_embeddings', initializer=self._cause_embeddings,
                                                 dtype=tf.float32)
    def apply(self, inputs):
        embedding_state = gen_array_ops.gather_v2(self._cause_embeddings, inputs, axis=0)
        return embedding_state

    def __call__(self, inputs):
        return self.apply(inputs)
