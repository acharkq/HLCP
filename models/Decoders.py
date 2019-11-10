import sys
import tensorflow as tf
from tensorflow.contrib import rnn
from tensorflow.python.layers.core import Dense
from tensorflow.contrib.seq2seq import dynamic_decode, tile_batch
from tensorflow_override.MyTrainingHelper import MyTrainingHelper
from tensorflow_override.MyAttentionWrapper import MyBahdanauAttention, MyAttentionWrapper
from tensorflow_override.MyBasicDecoder import MyBasicDecoder
from tensorflow_override.MyBeamSearchDecoder import MyBeamSearchDecoder
from tensorflow_override.my_rnn_cell import MyLayerNormBasicLSTMCell, LSTMStateTuple
from tensorflow.python.ops import array_ops


def NameDecoder(name, *args, **kw):
    return getattr(sys.modules[__name__], name)(*args, **kw)


class Decoder(object):
    _dropout_keep_prob = 1

    def __init__(self, params, mode):
        self._dropout_keep_prob = params['dropout_keep_prob'] if mode == tf.estimator.ModeKeys.TRAIN else 1


class HieDeocoder(Decoder):
    _beam_width = 1
    _SOS = -1
    _embedding_size = -1
    _max_cause_length = -1
    _EOS = -1
    _lstm_cell = None
    _initial_state = None
    _final_state = None
    _project_dense = None
    _cause_table = None
    _attention_values = None
    _attention_values_length = None
    _hie = True

    def __init__(self, cause_table, initial_state, final_state, params, mode):
        super(HieDeocoder, self).__init__(params, mode)
        self._initial_state = initial_state
        self._final_state = final_state
        self._beam_width = params['beam_width']
        self._SOS = params['SOS']
        self._embedding_size = params['embedding_size']
        self._project_dense = Dense(params['num_causes'], name='dense_for_decoder')
        self._cause_table = cause_table
        self._hie = params['hie']
        self._max_cause_length = params['max_cause_length']
        self._EOS = params['EOS']


class LSTM_Decoder(HieDeocoder):

    def __init__(self, cause_table, initial_state, final_state, params, mode):
        super(LSTM_Decoder, self).__init__(cause_table, initial_state, final_state, params, mode)
        self._lstm_cell = MyLayerNormBasicLSTMCell(num_units=params['num_units'],
                                                   dropout_keep_prob=self._dropout_keep_prob)

    def train(self, embedded_cause, cause_label, cause_length):
        batch_size = tf.shape(cause_label)[0]
        initial_state = LSTMStateTuple(self._initial_state, self._initial_state,
                                       last_choice=array_ops.fill([batch_size], self._SOS))
        train_helper = MyTrainingHelper(embedded_cause, cause_label, cause_length)
        train_decoder = MyBasicDecoder(self._lstm_cell, train_helper, initial_state, lookup_table=self._cause_table,
                                       output_layer=self._project_dense, hie=self._hie)
        decoder_output_train, decoder_state_train, decoder_len_train = dynamic_decode(train_decoder,
                                                                                      maximum_iterations=self._max_cause_length - 1,
                                                                                      parallel_iterations=128,
                                                                                      scope='decoder')
        return decoder_output_train, decoder_state_train, decoder_len_train

    def infer(self, cause_encoder, ):
        batch_size = tf.shape(self._initial_state)[0]
        tiled_initial_state = tile_batch(self._initial_state, multiplier=self._beam_width)
        tiled_initial_state = LSTMStateTuple(tiled_initial_state, tiled_initial_state,
                                             last_choice=array_ops.fill([batch_size * self._beam_width], self._SOS))
        infer_decoder = MyBeamSearchDecoder(self._lstm_cell, embedding=cause_encoder,
                                            start_tokens=tf.fill([batch_size], self._SOS),
                                            end_token=self._EOS, initial_state=tiled_initial_state,
                                            beam_width=self._beam_width,
                                            output_layer=self._project_dense, lookup_table=self._cause_table,
                                            length_penalty_weight=0.7, hie=self._hie)
        cause_output_infer, cause_state_infer, cause_length_infer = dynamic_decode(infer_decoder,
                                                                                   parallel_iterations=128,
                                                                                   maximum_iterations=self._max_cause_length - 1,
                                                                                   scope='decoder')
        return cause_output_infer, cause_state_infer, cause_length_infer


class LSTM_Attention(HieDeocoder):
    _batch_size = None

    def __init__(self, cause_table, initial_state, final_state, params, mode):
        super(LSTM_Attention, self).__init__(cause_table, initial_state, final_state, params, mode)
        self._lstm_cell = rnn.LayerNormBasicLSTMCell(num_units=params['num_units'],
                                                     dropout_keep_prob=self._dropout_keep_prob)

    def fill(self, attention_values, attention_values_length, ):
        self._batch_size = tf.shape(attention_values)[0]
        self._attention_values = attention_values
        self._attention_values_length = attention_values_length

    def train(self, embedded_cause, cause_label, cause_length, ):
        attention_mechanism = MyBahdanauAttention(num_units=self._embedding_size, memory=self._attention_values,
                                                  memory_sequence_length=self._attention_values_length)
        initial_state = rnn.LSTMStateTuple(self._initial_state, self._initial_state)
        cell = MyAttentionWrapper(self._lstm_cell, attention_mechanism, sot=self._SOS, output_attention=False,
                                  name='MyAttentionWrapper')
        cell_state = cell.zero_state(dtype=tf.float32, batch_size=self._batch_size)
        cell_state = cell_state.clone(cell_state=initial_state, attention=self._final_state)
        train_helper = MyTrainingHelper(embedded_cause, cause_label, cause_length)
        train_decoder = MyBasicDecoder(cell, train_helper, cell_state, lookup_table=self._cause_table,
                                       output_layer=self._project_dense, hie=self._hie)
        decoder_output_train, decoder_state_train, decoder_len_train = dynamic_decode(train_decoder,
                                                                                      maximum_iterations=self._max_cause_length - 1,
                                                                                      parallel_iterations=128,
                                                                                      scope='decoder')
        return decoder_output_train, decoder_state_train, decoder_len_train

    def infer(self, cause_encoder, ):
        tiled_memory = tile_batch(self._attention_values, multiplier=self._beam_width)
        tiled_memory_sequence = tile_batch(self._attention_values_length, multiplier=self._beam_width)
        tiled_initial_state = tile_batch(self._initial_state, multiplier=self._beam_width)
        tiled_initial_state = rnn.LSTMStateTuple(tiled_initial_state, tiled_initial_state)
        tiled_first_attention = tile_batch(self._final_state, multiplier=self._beam_width)
        attention_mechanism = MyBahdanauAttention(num_units=self._embedding_size, memory=tiled_memory,
                                                  memory_sequence_length=tiled_memory_sequence)
        cell = MyAttentionWrapper(self._lstm_cell, attention_mechanism, sot=self._SOS, output_attention=False,
                                  name='MyAttentionWrapper')
        cell_state = cell.zero_state(dtype=tf.float32, batch_size=self._batch_size * self._beam_width)
        cell_state = cell_state.clone(cell_state=tiled_initial_state, attention=tiled_first_attention)
        infer_decoder = MyBeamSearchDecoder(cell, embedding=cause_encoder,
                                            start_tokens=tf.fill([self._batch_size], self._SOS),
                                            end_token=self._EOS, initial_state=cell_state, beam_width=self._beam_width,
                                            output_layer=self._project_dense, lookup_table=self._cause_table,
                                            length_penalty_weight=0.7, hie=self._hie)
        cause_output_infer, cause_state_infer, cause_length_infer = dynamic_decode(infer_decoder,
                                                                                   parallel_iterations=128,
                                                                                   maximum_iterations=self._max_cause_length - 1,
                                                                                   scope='decoder')
        return cause_output_infer, cause_state_infer, cause_length_infer
