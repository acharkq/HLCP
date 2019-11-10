import time
import tensorflow as tf
import numpy as np
from tensorflow.python.layers.core import Dense
from tensorflow.contrib import rnn
from tensorflow.contrib.seq2seq import sequence_loss, dynamic_decode, \
    tile_batch, BahdanauAttention, AttentionWrapper, GreedyEmbeddingHelper
from tensorflow_override.MyBeamSearchDecoder import MyBeamSearchDecoder
from tensorflow_override.MyAttentionWrapper import MyAttentionWrapper, MyBahdanauAttention, \
    MyAttentionWrapper_v2
from tensorflow_override.MyBasicDecoder import MyBasicDecoder
from tensorflow_override.MyTrainingHelper import MyTrainingHelper
from tensorflow.python.ops import gen_array_ops
from models.Encoders import EncoderOutput


def masked_mean_pooling(input, sequence_masks, max_sequence_length):
    '''do masked mean pooling on the sequence_length dimension of a tensor with shape=[batch_size, sequence_length, feature]'''
    sequence_mask = tf.sequence_mask(sequence_masks, maxlen=max_sequence_length, dtype=tf.float32)
    sequence_mask = tf.expand_dims(sequence_mask, axis=2)
    tmp = tf.multiply(input, sequence_mask) / tf.reduce_sum(sequence_mask, axis=1)[:, tf.newaxis]
    pooled = tf.reduce_sum(tmp, axis=1)
    return pooled


def _compute_loss(logits, target_output, target_weights, batch_size):
    """Compute optimization loss."""
    """logits shape=[batch_size, num_steps, num_classes]"""
    """思路 tf.logical_and(target_weights, new_weight)"""
    loss = sequence_loss(logits=logits, targets=target_output[:, 1:], weights=target_weights)
    return loss


def load_embedding(model, vocab_size, embedding_size):
    if isinstance(model, np.ndarray):
        embeddings_word = tf.get_variable(initializer=model, trainable=True, name='word_embeddings')
    elif model:
        embedding_matrix = np.zeros((vocab_size, embedding_size), dtype=np.float32)
        for i in range(vocab_size):
            embedding_vector = model.wv[model.wv.index2word[i]]
            embedding_matrix[i] = embedding_vector
        embeddings_word = tf.constant(embedding_matrix)
        embeddings_word = tf.get_variable(initializer=embeddings_word, trainable=True, name='word_embeddings')
    else:
        embeddings_word = tf.get_variable(name='embeddings_word',
                                          initializer=tf.random_uniform([vocab_size, embedding_size], -1.0, 1.0))
    return embeddings_word


def cnn_encoder(inputs, sequence_length, params, mode):
    '''simple cnn encoder'''
    # convolutional layer
    expanded_input = tf.expand_dims(inputs, -1)
    conv = tf.layers.conv2d(
        inputs=expanded_input,
        filters=params['num_filters'],
        kernel_size=[params['kernel_size'], params['embedding_size']],
        padding='valid',
        activation=tf.nn.relu, name='cnn_encoder')

    # conv shape=[-1,max_sequence_length - kernel_size + 1, 1, num_filter]
    conv = tf.layers.dropout(conv, 0.5, training=mode == tf.estimator.ModeKeys.TRAIN)
    conv = tf.squeeze(conv, axis=2)
    max_pooled = tf.layers.max_pooling1d(conv, params['max_sequence_length'] - params['kernel_size'] + 1,
                                         strides=1)
    max_pooled = tf.squeeze(max_pooled, axis=1, name='max_pooled')
    # calculate the output length of convolution layer
    conv_length = sequence_length - params['kernel_size'] + 1

    # manual masked mean pooling
    first_attention = masked_mean_pooling(conv, conv_length,
                                          params['max_sequence_length'] - params['kernel_size'] + 1)
    return EncoderOutput(outputs=conv, final_state=first_attention, attention_values=conv,
                         attention_values_length=conv_length, initial_state=max_pooled)


def multi_cnn_encoder(inputs, sequence_length, params, mode):
    pooled_outputs = []
    conv_outputs = []
    inputs = tf.expand_dims(inputs, -1)
    dropout_keep_prob = params['dropout_keep_prob'] if mode == tf.estimator.ModeKeys.TRAIN else 1
    for i, filter_size in enumerate(params['filter_sizes']):
        with tf.name_scope('conv-maxpool-%s' % filter_size):
            conv = tf.layers.conv2d(inputs, params['num_filter'], [filter_size, params['embedding_size']],
                                    activation=tf.nn.relu, name='conv-size-%s' % filter_size, reuse=False,
                                    padding='same', strides=[1, params['embedding_size']])
            conv_outputs.append(conv)
            pooled = tf.layers.max_pooling2d(conv, pool_size=[params['max_sequence_length'], 1],
                                             strides=[1, 1], name='pool')
            # pooled shape = [batch_size, .., num_filters]
            pooled_outputs.append(pooled)
    h_pool = tf.concat(pooled_outputs, axis=3)
    h_pool = tf.squeeze(h_pool, [1, 2])
    h_conv = tf.concat(conv_outputs, axis=3)
    h_conv = tf.squeeze(h_conv, 2)
    with tf.name_scope('drop_out'):
        h_pool = tf.nn.dropout(h_pool, dropout_keep_prob)
        h_conv = tf.nn.dropout(h_conv, dropout_keep_prob)
    first_attention = masked_mean_pooling(h_conv, sequence_length, params['max_sequence_length'])
    return EncoderOutput(outputs=h_conv, final_state=first_attention, attention_values=h_conv,
                         attention_values_length=sequence_length, initial_state=h_pool)


def encoders(inputs, sequence_length, params, mode):
    if params['encoder'] == 'cnn':
        return cnn_encoder(inputs, sequence_length, params, mode)
    elif params['encoder'] == 'multi_cnn':
        return multi_cnn_encoder(inputs, sequence_length, params, mode)


class CauseEncoder(object):
    ''''updated on 2018-7-28 for cause buchang'''

    def __init__(self, word_embeddings, params):
        self._word_embeddings = word_embeddings
        self._params = params
        self._embedding_size = int(self._word_embeddings.shape[1])
        # self._lstm_cell = rnn.BasicLSTMCell(num_units=self._embedding_size, reuse=tf.AUTO_REUSE)
        self._lstm_cell = rnn.LayerNormBasicLSTMCell(num_units=self._embedding_size, reuse=tf.AUTO_REUSE)
        self._cause_id_table = tf.constant(params['cause_id_table'], name='cause_id_table')
        self._cause_id_table_length = tf.constant(params['cause_id_table_length'], name='cause_id_table_length')

    def __call__(self, inputs):
        return self.apply(inputs)

    def apply(self, inputs):
        '''
        inputs shape = [batch_size, max_num_causes] the word id of input cause
        sequence_length = [batch_size,]
        return shape = [batch_size, max_cause_num, embedding_size]
        '''
        max_num_causes = int(inputs.shape[1])
        max_word_num = int(self._cause_id_table.shape[1])
        inputs_word_ids = gen_array_ops.gather_v2(self._cause_id_table, inputs, axis=0)
        # shape = [batch_size, max_num_causes, max_words_length]
        embedded_inputs = gen_array_ops.gather_v2(self._word_embeddings, inputs_word_ids, axis=0)
        # shape = [batch_size, max_num_causes, max_words_length, embedding_size], and it will be flattend
        embedded_inputs = tf.reshape(embedded_inputs, [-1, max_word_num, self._embedding_size])
        inputs_word_length = gen_array_ops.gather_v2(self._cause_id_table_length, inputs, axis=0)
        # shape = [batch_size, max_num_causes]
        inputs_word_length = tf.reshape(inputs_word_length, [-1])
        lstm_zero_state = self._lstm_cell.zero_state(tf.shape(inputs_word_length)[0], tf.float32)
        outputs, state = tf.nn.dynamic_rnn(self._lstm_cell, embedded_inputs, inputs_word_length, lstm_zero_state,
                                           scope='lstm_cause_encoder')
        # state = [c, h] shape = [batch_size * max_num_causes, embedding_size]
        state = tf.reshape(state[0], [-1, max_num_causes, self._embedding_size])
        return state


def _match_model_fn(features, labels, mode, params):
    '''the model function for the custom estimator, consists of text extraction(realized with cnn) and hierarchical multilabel'''
    '''set parameters'''
    with tf.device('/gpu:0'), tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
        # set hyper parameters
        embedding_size = params['embedding_size']
        num_units = params['num_units']
        if mode == tf.estimator.ModeKeys.TRAIN:
            dropout_keep_prob = params['dropout_keep_prob']
        else:
            dropout_keep_prob = 1
        beam_width = params['beam_width']
        EOS = params['EOS']
        SOS = params['SOS']
        # set training parameters
        max_sequence_length = params['max_sequence_length']
        max_cause_length = params['max_cause_length']
        vocab_size = params['vocab_size']
        num_causes = EOS + 1
        '''process input and target'''
        # input layer
        input = tf.reshape(features['content'], [-1, max_sequence_length])
        batch_size = tf.shape(input)[0]
        input_length = tf.reshape(features['content_length'], [batch_size])
        cause_label = tf.reshape(labels['cause_label'], [batch_size, max_cause_length])
        cause_length = tf.reshape(labels['cause_length'], [batch_size])

        # necessary cast
        input = tf.cast(input, dtype=tf.int32)
        input_length = tf.cast(input_length, dtype=tf.int32)
        cause_label = tf.cast(cause_label, dtype=tf.int32)
        cause_length = tf.cast(cause_length, dtype=tf.int32)

        # word embedding layer
        embeddings_word = load_embedding(params['word2vec_model'], vocab_size, embedding_size)

        embedded_input = gen_array_ops.gather_v2(embeddings_word, input, axis=0)
        # cause-label embedding layer
        embeddings_cause = tf.get_variable('cause_embeddings', shape=[num_causes, embedding_size],
                                           initializer=tf.random_normal_initializer(mean=0.0, stddev=0.1))
        # num_cause + 1 for the EOS
        embedded_cause = gen_array_ops.gather_v2(embeddings_cause, cause_label, axis=0)
        # cause lookpu_table
        cause_table = tf.constant(params['cause_table'], dtype=tf.int32)
        encoder_output = encoders(embedded_input, input_length, params, mode)

        '''hierarchical multilabel decoder'''
        # build lstm cell with attention
        lstm = rnn.LayerNormBasicLSTMCell(num_units=num_units, dropout_keep_prob=dropout_keep_prob, reuse=tf.AUTO_REUSE)
        # the subtraction at the end of the line is a ele-wise subtraction supported by tensorflow

        attention_mechanism = MyBahdanauAttention(num_units=embedding_size, memory=encoder_output.attention_values,
                                                  memory_sequence_length=encoder_output.attention_values_length)
        initial_state = rnn.LSTMStateTuple(encoder_output.initial_state, encoder_output.initial_state)
        cell = MyAttentionWrapper(lstm, attention_mechanism, sot=SOS, output_attention=False,
                                  name="MyAttentionWrapper")
        cell_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        cell_state = cell_state.clone(cell_state=initial_state, attention=encoder_output.final_state)

        # extra dense layer to project a rnn output into a classification
        project_dense = Dense(num_causes, _reuse=tf.AUTO_REUSE, _scope='project_dense_scope', name='project_dense')

        # train_decoder
        train_helper = MyTrainingHelper(embedded_cause, cause_label, cause_length)
        train_decoder = MyBasicDecoder(cell, train_helper, cell_state, lookup_table=cause_table,
                                       output_layer=project_dense, hie=params['hie'])

        decoder_output_train, decoder_state_train, decoder_len_train = dynamic_decode(train_decoder,
                                                                                      maximum_iterations=max_cause_length - 1,
                                                                                      parallel_iterations=64,
                                                                                      scope='decoder')

        tiled_memory_sequence_length = tile_batch(encoder_output.attention_values_length, multiplier=beam_width)
        tiled_memory = tile_batch(encoder_output.attention_values, multiplier=beam_width)
        tiled_encoder_output_initital_state = tile_batch(encoder_output.initial_state, multiplier=beam_width)
        tiled_initial_state = rnn.LSTMStateTuple(tiled_encoder_output_initital_state,
                                                 tiled_encoder_output_initital_state)
        tiled_first_attention = tile_batch(encoder_output.final_state, multiplier=beam_width)

        attention_mechanism = MyBahdanauAttention(num_units=embedding_size, memory=tiled_memory,
                                                  memory_sequence_length=tiled_memory_sequence_length)

        cell = MyAttentionWrapper(lstm, attention_mechanism, sot=SOS, output_attention=False,
                                  name="MyAttentionWrapper")
        cell_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size * beam_width)
        cell_state = cell_state.clone(cell_state=tiled_initial_state, attention=tiled_first_attention)
        infer_decoder = MyBeamSearchDecoder(cell, embedding=embeddings_cause, sots=tf.fill([batch_size], SOS),
                                            start_tokens=tf.fill([batch_size], SOS),
                                            end_token=EOS, initial_state=cell_state, beam_width=beam_width,
                                            output_layer=project_dense, lookup_table=cause_table, hie=params['hie'],
                                            length_penalty_weight=0.7)

        cause_output_infer, cause_state_infer, cause_length_infer = dynamic_decode(infer_decoder,
                                                                                   parallel_iterations=64,
                                                                                   maximum_iterations=max_cause_length - 1,
                                                                                   scope='decoder')

        # loss
        mask_for_cause = tf.sequence_mask(cause_length, max_cause_length - 1, dtype=tf.float32)
        # loss = sequence_loss(logits=padded_train_output, targets=cause_label, weights=mask_for_cause, name='loss')
        tmp_padding = tf.pad(decoder_output_train.rnn_output,
                             [[0, 0], [0, max_cause_length - 1 - tf.shape(decoder_output_train.rnn_output)[1]], [0, 0]],
                             constant_values=0)

        loss = _compute_loss(tmp_padding, cause_label, mask_for_cause, batch_size)
        # predicted_ids: [batch_size, max_cause_length, beam_width]

        predicted_and_cause_ids = tf.transpose(cause_output_infer.predicted_ids, perm=[0, 2, 1],
                                               name='predicted_cause_ids')

        # for monitoring
        cause_label_expanded = tf.reshape(cause_label[:, 1:], [-1, 1, max_cause_length - 1])
        predicted_and_cause_ids = tf.pad(predicted_and_cause_ids,
                                         [[0, 0], [0, 0],
                                          [0, max_cause_length - 1 - tf.shape(predicted_and_cause_ids)[2]]],
                                         constant_values=EOS)
        predicted_and_cause_ids = tf.concat([predicted_and_cause_ids, cause_label_expanded], axis=1,
                                            name='predicted_and_cause_ids')
        predicted_and_cause_ids = tf.reshape(predicted_and_cause_ids, [-1, beam_width + 1, max_cause_length - 1])
        predicted_and_cause_ids_train = tf.concat([decoder_output_train.sample_id, cause_label[:, 1:]], axis=1,
                                                  name='predicted_and_cause_ids_train')
        predictions = {
            'predicted_and_cause_ids': predicted_and_cause_ids,
        }

        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # warm_up_constant = params['warm_up_steps'] ** (-1.5)
            # embedding_constant = embedding_size ** (-0.5)
            # global_step = tf.to_float(tf.train.get_global_step())
            # learning_rate = tf.minimum(1 / tf.sqrt(global_step),
            #                            warm_up_constant * global_step) * embedding_constant
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9)
            optimizer = tf.train.AdamOptimizer()
            # # train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            # '''using gradient clipping'''
            # loss = tf.Print(loss, [loss, 'to be clear, this is the loss'])
            grads_and_vars = optimizer.compute_gradients(loss)
            clipped_gvs = [ele if ele[0] is None else (tf.clip_by_value(ele[0], -0.1, 0.1), ele[1]) for ele in
                           grads_and_vars]
            train_op = optimizer.apply_gradients(clipped_gvs, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # predicted_cause_ids shape = [batch_size, cause_length]
        # cause_label = [batch_size, cause_length]
        # 　select the predicted cause with the highest possibility
        # todo: evalutaion
        # bi_predicted_cause_ids = binarizer(predicted_cause_ids[:, 0, :], num_causes)
        # bi_cause_label = binarizer(cause_label, num_causes)

        # todo: now I have to leave the evaluation work be done outside the estimator
        eval_metric_ops = {
            'predicted_and_cause_ids': tf.contrib.metrics.streaming_concat(predicted_and_cause_ids),
            # 'precision': tf.metrics.precision(bi_cause_label, bi_predicted_cause_ids),
            # 'recall': tf.metrics.recall(bi_cause_label, bi_predicted_cause_ids),
            # 'f1-score': f_score(bi_cause_label, bi_predicted_cause_ids),
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def _match_model_fn_v5(features, labels, mode, params):
    '''this version uses origianl seq2seq, but uses a lstm merges the cause and word embedding_tabel'''
    '''set parameters'''
    with tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
        # set hyper parameters
        embedding_size = params['embedding_size']
        num_units = params['num_units']
        if mode == tf.estimator.ModeKeys.TRAIN:
            dropout_keep_prob = params['dropout_keep_prob']
        else:
            dropout_keep_prob = 1
        beam_width = params['beam_width']
        EOS = params['EOS']
        SOS = params['SOS']
        # set training parameters
        max_sequence_length = params['max_sequence_length']
        max_cause_length = params['max_cause_length']
        vocab_size = params['vocab_size']
        num_causes = EOS + 1
        '''process input and target'''
        # input layer
        input = tf.reshape(features['content'], [-1, max_sequence_length])
        batch_size = tf.shape(input)[0]
        input_length = tf.reshape(features['content_length'], [batch_size])
        cause_label = tf.reshape(labels['cause_label'], [batch_size, max_cause_length])
        cause_length = tf.reshape(labels['cause_length'], [batch_size])

        # necessary cast
        input = tf.cast(input, dtype=tf.int32)
        input_length = tf.cast(input_length, dtype=tf.int32)
        cause_label = tf.cast(cause_label, dtype=tf.int32)
        cause_length = tf.cast(cause_length, dtype=tf.int32)
        # word embedding layer
        embeddings_word = load_embedding(params['word2vec_model'], vocab_size, embedding_size)

        embedded_input = gen_array_ops.gather_v2(embeddings_word, input, axis=0)
        # cause-label embedding layer
        cause_encoder = CauseEncoder(word_embeddings=embeddings_word, params=params)
        embedded_cause = cause_encoder.apply(cause_label)

        # cause lookpu_table
        cause_table = tf.constant(params['cause_table'], dtype=tf.int32)
        encoder_output = encoders(embedded_input, input_length, params, mode)

        '''hierarchical multilabel decoder'''
        # build lstm cell with attention
        lstm = rnn.LayerNormBasicLSTMCell(num_units=num_units, reuse=tf.AUTO_REUSE, dropout_keep_prob=dropout_keep_prob)
        # lstm = rnn.DropoutWrapper(lstm, output_keep_prob=dropout_keep_prob)
        # the subtraction at the end of the line is a ele-wise subtraction supported by tensorflow

        attention_mechanism = MyBahdanauAttention(num_units=embedding_size, memory=encoder_output.attention_values,
                                                  memory_sequence_length=encoder_output.attention_values_length)
        initial_state = rnn.LSTMStateTuple(encoder_output.initial_state, encoder_output.initial_state)
        cell = MyAttentionWrapper(lstm, attention_mechanism, sot=SOS, output_attention=False,
                                  name="MyAttentionWrapper")
        cell_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        cell_state = cell_state.clone(cell_state=initial_state, attention=encoder_output.final_state)

        # extra dense layer to project a rnn output into a classification
        project_dense = Dense(num_causes, _reuse=tf.AUTO_REUSE, _scope='project_dense_scope', name='project_dense')

        # train_decoder
        train_helper = MyTrainingHelper(embedded_cause, cause_label, cause_length)
        train_decoder = MyBasicDecoder(cell, train_helper, cell_state, lookup_table=cause_table,
                                       output_layer=project_dense, hie=params['hie'])

        decoder_output_train, decoder_state_train, decoder_len_train = dynamic_decode(train_decoder,
                                                                                      maximum_iterations=max_cause_length - 1,
                                                                                      parallel_iterations=64,
                                                                                      scope='decoder')

        # beam_width = 1
        tiled_memory_sequence_length = tile_batch(encoder_output.attention_values_length, multiplier=beam_width)
        tiled_memory = tile_batch(encoder_output.attention_values, multiplier=beam_width)
        tiled_encoder_output_initital_state = tile_batch(encoder_output.initial_state, multiplier=beam_width)
        tiled_initial_state = rnn.LSTMStateTuple(tiled_encoder_output_initital_state,
                                                 tiled_encoder_output_initital_state)
        tiled_first_attention = tile_batch(encoder_output.final_state, multiplier=beam_width)

        attention_mechanism = MyBahdanauAttention(num_units=embedding_size, memory=tiled_memory,
                                                  memory_sequence_length=tiled_memory_sequence_length)

        cell = MyAttentionWrapper(lstm, attention_mechanism, sot=SOS, output_attention=False,
                                  name="MyAttentionWrapper")
        cell_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size * beam_width)
        cell_state = cell_state.clone(cell_state=tiled_initial_state, attention=tiled_first_attention)
        infer_decoder = MyBeamSearchDecoder(cell, embedding=cause_encoder, sots=tf.fill([batch_size], SOS),
                                            start_tokens=tf.fill([batch_size], SOS),
                                            end_token=EOS, initial_state=cell_state, beam_width=beam_width,
                                            output_layer=project_dense, lookup_table=cause_table,
                                            length_penalty_weight=0.7, hie=params['hie'])

        cause_output_infer, cause_state_infer, cause_length_infer = dynamic_decode(infer_decoder,
                                                                                   parallel_iterations=64,
                                                                                   maximum_iterations=max_cause_length - 1,
                                                                                   scope='decoder')

        # loss
        mask_for_cause = tf.sequence_mask(cause_length - 1, max_cause_length - 1, dtype=tf.float32)
        tmp_padding = tf.pad(decoder_output_train.rnn_output,
                             [[0, 0], [0, max_cause_length - 1 - tf.shape(decoder_output_train.rnn_output)[1]], [0, 0]],
                             constant_values=0)
        loss = _compute_loss(tmp_padding, cause_label, mask_for_cause, batch_size)
        # loss = tf.Print(loss, [loss, 'losss'], summarize=1)
        # predicted_ids: [batch_size, max_cause_length, beam_width]

        predicted_and_cause_ids = tf.transpose(cause_output_infer.predicted_ids, perm=[0, 2, 1],
                                               name='predicted_cause_ids')

        # for monitoring
        cause_label_expanded = tf.reshape(cause_label[:, 1:], [-1, 1, max_cause_length - 1])
        predicted_and_cause_ids = tf.pad(predicted_and_cause_ids,
                                         [[0, 0], [0, 0],
                                          [0, max_cause_length - 1 - tf.shape(predicted_and_cause_ids)[2]]],
                                         constant_values=EOS)
        predicted_and_cause_ids = tf.concat([predicted_and_cause_ids, cause_label_expanded], axis=1,
                                            name='predicted_and_cause_ids')
        predicted_and_cause_ids = tf.reshape(predicted_and_cause_ids, [-1, beam_width + 1, max_cause_length - 1])
        predicted_and_cause_ids_train = tf.concat([decoder_output_train.sample_id, cause_label[:, 1:]], axis=1,
                                                  name='predicted_and_cause_ids_train')
        predictions = {
            'predicted_and_cause_ids': predicted_and_cause_ids,
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # warm_up_constant = params['warm_up_steps'] ** (-1.5)
            # embedding_constant = embedding_size ** (-0.5)
            # global_step = tf.to_float(tf.train.get_global_step())
            # learning_rate = tf.minimum(1 / tf.sqrt(global_step),
            #                            warm_up_constant * global_step) * embedding_constant
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9)
            optimizer = tf.train.AdamOptimizer()
            # # train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            # '''using gradient clipping'''
            # loss = tf.Print(loss, [loss, 'to be clear, this is the loss'])
            grads_and_vars = optimizer.compute_gradients(loss)
            clipped_gvs = [ele if ele[0] is None else (tf.clip_by_value(ele[0], -0.1, 0.1), ele[1]) for ele in
                           grads_and_vars]
            train_op = optimizer.apply_gradients(clipped_gvs, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # predicted_cause_ids shape = [batch_size, cause_length]
        # cause_label = [batch_size, cause_length]
        # 　select the predicted cause with the highest possibility
        # todo: evalutaion
        # bi_predicted_cause_ids = binarizer(predicted_cause_ids[:, 0, :], num_causes)
        # bi_cause_label = binarizer(cause_label, num_causes)

        # todo: now I have to leave the evaluation work be done outside the estimator
        eval_metric_ops = {
            'predicted_and_cause_ids': tf.contrib.metrics.streaming_concat(predicted_and_cause_ids),
            # 'precision': tf.metrics.precision(bi_cause_label, bi_predicted_cause_ids),
            # 'recall': tf.metrics.recall(bi_cause_label, bi_predicted_cause_ids),
            # 'f1-score': f_score(bi_cause_label, bi_predicted_cause_ids),
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)


def _match_model_fn_v6(features, labels, mode, params):
    '''
    this version uses origianl seq2seq, but uses a lstm merges the cause and word embedding_tabel

    and this version use the input embedding as the attention query
    '''
    # print('aaa')
    '''set parameters'''
    with tf.device('/gpu:0'), tf.variable_scope('model', reuse=tf.AUTO_REUSE) as scope:
        # set hyper parameters
        embedding_size = params['embedding_size']
        num_units = params['num_units']
        if mode == tf.estimator.ModeKeys.TRAIN:
            dropout_keep_prob = params['dropout_keep_prob']
        else:
            dropout_keep_prob = 1
        beam_width = params['beam_width']
        EOS = params['EOS']
        SOS = params['SOS']
        # set training parameters
        max_sequence_length = params['max_sequence_length']
        max_cause_length = params['max_cause_length']
        vocab_size = params['vocab_size']
        num_causes = EOS + 1
        '''process input and target'''
        # input layer
        input = tf.reshape(features['content'], [-1, max_sequence_length])
        batch_size = tf.shape(input)[0]
        input_length = tf.reshape(features['content_length'], [batch_size])
        cause_label = tf.reshape(labels['cause_label'], [batch_size, max_cause_length])
        cause_length = tf.reshape(labels['cause_length'], [batch_size])

        # necessary cast
        input = tf.cast(input, dtype=tf.int32)
        input_length = tf.cast(input_length, dtype=tf.int32)
        cause_label = tf.cast(cause_label, dtype=tf.int32)
        cause_length = tf.cast(cause_length, dtype=tf.int32)

        # word embedding layer
        embeddings_word = load_embedding(params['word2vec_model'], vocab_size, embedding_size)

        embedded_input = gen_array_ops.gather_v2(embeddings_word, input, axis=0)
        # cause-label embedding layer
        cause_encoder = CauseEncoder(word_embeddings=embeddings_word, params=params)
        embedded_cause = cause_encoder.apply(cause_label)

        # cause lookpu_table
        cause_table = tf.constant(params['cause_table'], dtype=tf.int32)
        encoder_output = encoders(embedded_input, input_length, params, mode)

        '''hierarchical multilabel decoder'''
        # build lstm cell with attention
        lstm = rnn.LayerNormBasicLSTMCell(num_units=num_units, reuse=tf.AUTO_REUSE, dropout_keep_prob=dropout_keep_prob)
        # lstm = rnn.DropoutWrapper(lstm, output_keep_prob=dropout_keep_prob)
        # the subtraction at the end of the line is a ele-wise subtraction supported by tensorflow

        attention_mechanism = MyBahdanauAttention(num_units=embedding_size, memory=encoder_output.attention_values,
                                                  memory_sequence_length=encoder_output.attention_values_length)
        initial_state = rnn.LSTMStateTuple(encoder_output.initial_state, encoder_output.initial_state)
        cell = MyAttentionWrapper_v2(lstm, attention_mechanism, sot=SOS, output_attention=False,
                                     name="MyAttentionWrapper")
        cell_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size)
        cell_state = cell_state.clone(cell_state=initial_state, attention=encoder_output.final_state)

        # extra dense layer to project a rnn output into a classification
        project_dense = Dense(num_causes, _reuse=tf.AUTO_REUSE, _scope='project_dense_scope', name='project_dense')

        # train_decoder
        train_helper = MyTrainingHelper(embedded_cause, cause_label, cause_length)
        train_decoder = MyBasicDecoder(cell, train_helper, cell_state, lookup_table=cause_table,
                                       output_layer=project_dense, hie=params['hie'])

        decoder_output_train, decoder_state_train, decoder_len_train = dynamic_decode(train_decoder,
                                                                                      maximum_iterations=max_cause_length - 1,
                                                                                      parallel_iterations=64,
                                                                                      scope='decoder')

        # beam_width = 1
        tiled_memory_sequence_length = tile_batch(encoder_output.attention_values_length, multiplier=beam_width)
        tiled_memory = tile_batch(encoder_output.attention_values, multiplier=beam_width)
        tiled_encoder_output_initital_state = tile_batch(encoder_output.initial_state, multiplier=beam_width)
        tiled_initial_state = rnn.LSTMStateTuple(tiled_encoder_output_initital_state,
                                                 tiled_encoder_output_initital_state)
        tiled_first_attention = tile_batch(encoder_output.final_state, multiplier=beam_width)

        attention_mechanism = MyBahdanauAttention(num_units=embedding_size, memory=tiled_memory,
                                                  memory_sequence_length=tiled_memory_sequence_length)

        cell = MyAttentionWrapper_v2(lstm, attention_mechanism, sot=SOS, output_attention=False,
                                     name="MyAttentionWrapper")
        cell_state = cell.zero_state(dtype=tf.float32, batch_size=batch_size * beam_width)
        cell_state = cell_state.clone(cell_state=tiled_initial_state, attention=tiled_first_attention)
        infer_decoder = MyBeamSearchDecoder(cell, embedding=cause_encoder, sots=tf.fill([batch_size], SOS),
                                            start_tokens=tf.fill([batch_size], SOS),
                                            end_token=EOS, initial_state=cell_state, beam_width=beam_width,
                                            output_layer=project_dense, lookup_table=cause_table,
                                            length_penalty_weight=0.7, hie=params['hie'])

        cause_output_infer, cause_state_infer, cause_length_infer = dynamic_decode(infer_decoder,
                                                                                   parallel_iterations=64,
                                                                                   maximum_iterations=max_cause_length - 1,
                                                                                   scope='decoder')

        # loss
        mask_for_cause = tf.sequence_mask(cause_length - 1, max_cause_length - 1, dtype=tf.float32)
        # loss = sequence_loss(logits=padded_train_output, targets=cause_label, weights=mask_for_cause, name='loss')
        tmp_padding = tf.pad(decoder_output_train.rnn_output,
                             [[0, 0], [0, max_cause_length - 1 - tf.shape(decoder_output_train.rnn_output)[1]], [0, 0]],
                             constant_values=0)
        loss = _compute_loss(tmp_padding, cause_label, mask_for_cause, batch_size)
        # predicted_ids: [batch_size, max_cause_length, beam_width]

        predicted_and_cause_ids = tf.transpose(cause_output_infer.predicted_ids, perm=[0, 2, 1],
                                               name='predicted_cause_ids')

        # for monitoring
        cause_label_expanded = tf.reshape(cause_label[:, 1:], [-1, 1, max_cause_length - 1])
        predicted_and_cause_ids = tf.pad(predicted_and_cause_ids,
                                         [[0, 0], [0, 0],
                                          [0, max_cause_length - 1 - tf.shape(predicted_and_cause_ids)[2]]],
                                         constant_values=EOS)
        predicted_and_cause_ids = tf.concat([predicted_and_cause_ids, cause_label_expanded], axis=1,
                                            name='predicted_and_cause_ids')
        predicted_and_cause_ids = tf.reshape(predicted_and_cause_ids, [-1, beam_width + 1, max_cause_length - 1])
        predicted_and_cause_ids_train = tf.concat([decoder_output_train.sample_id, cause_label[:, 1:]], axis=1,
                                                  name='predicted_and_cause_ids_train')

        predictions = {
            'predicted_and_cause_ids': predicted_and_cause_ids,
        }
        if mode == tf.estimator.ModeKeys.PREDICT:
            return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)

        if mode == tf.estimator.ModeKeys.TRAIN:
            # warm_up_constant = params['warm_up_steps'] ** (-1.5)
            # embedding_constant = embedding_size ** (-0.5)
            # global_step = tf.to_float(tf.train.get_global_step())
            # learning_rate = tf.minimum(1 / tf.sqrt(global_step),
            #                            warm_up_constant * global_step) * embedding_constant
            # optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate, beta1=0.9, beta2=0.98, epsilon=1e-9)
            optimizer = tf.train.AdamOptimizer()
            # # train_op = optimizer.minimize(loss=loss, global_step=tf.train.get_global_step())
            # '''using gradient clipping'''
            # loss = tf.Print(loss, [loss, 'to be clear, this is the loss'])
            grads_and_vars = optimizer.compute_gradients(loss)
            clipped_gvs = [ele if ele[0] is None else (tf.clip_by_value(ele[0], -0.1, 0.1), ele[1]) for ele in
                           grads_and_vars]
            train_op = optimizer.apply_gradients(clipped_gvs, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # predicted_cause_ids shape = [batch_size, cause_length]
        # cause_label = [batch_size, cause_length]
        # 　select the predicted cause with the highest possibility
        # todo: evalutaion
        # bi_predicted_cause_ids = binarizer(predicted_cause_ids[:, 0, :], num_causes)
        # bi_cause_label = binarizer(cause_label, num_causes)

        # todo: now I have to leave the evaluation work be done outside the estimator
        eval_metric_ops = {
            'predicted_and_cause_ids': tf.contrib.metrics.streaming_concat(predicted_and_cause_ids),
            # 'precision': tf.metrics.precision(bi_cause_label, bi_predicted_cause_ids),
            # 'recall': tf.metrics.recall(bi_cause_label, bi_predicted_cause_ids),
            # 'f1-score': f_score(bi_cause_label, bi_predicted_cause_ids),
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)
