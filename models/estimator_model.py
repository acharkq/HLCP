import tensorflow as tf
from tensorflow.contrib.seq2seq import sequence_loss
from models.EmbeddingLayer import EmbeddingLayer
from models.Encoders import TextEncoder
from models.Decoders import NameDecoder
from models.CauseEncoder import NameEncoder


def compute_loss(logits, target_output, target_weight):
    '''
    sequence loss
    as the decoder generate the sos symbol
    we removed it here
    '''
    loss = sequence_loss(logits=logits, targets=target_output[:, 1:], weights=target_weight)
    return loss


def estimator_model_1(features, labels, mode, params):
    '''
    this version uses origianl seq2seq, but uses a lstm merges the cause and word embedding_tabel

    and this version use the input embedding as the attention query
    '''
    # batch_size = params['batch_size']
    with tf.device(params['device']), tf.variable_scope('model', reuse=tf.AUTO_REUSE):
        input = tf.reshape(features['contents'], [-1, params['sen_len']])
        input_length = tf.reshape(features['content_lengths'], [-1])
        cause_label = tf.reshape(labels['causes'], [-1, params['max_cause_length']])
        cause_length = tf.reshape(labels['cause_lengths'], [-1])
        embedding_layer = EmbeddingLayer(params['word_embeddings'])

        cause_encoder = NameEncoder(name=params['name_encoder'], word_embeddings=embedding_layer, params=params)

        embedded_cause = cause_encoder(cause_label)
        input_embeddings = embedding_layer(input)
        encoder = TextEncoder(params['text_encoder'], params, mode)
        encoder_output = encoder(input_embeddings, input_length)
        # print(encoder_output.initial_state.shape)
        name_decoder = NameDecoder(name=params['name_decoder'], cause_table=params['cause_table'],
                                   initial_state=encoder_output.initial_state,
                                   final_state=encoder_output.final_state, params=params, mode=mode)
        if params['name_decoder'] == 'LSTM_Attention':
            name_decoder.fill(attention_values=encoder_output.attention_values,
                              attention_values_length=encoder_output.attention_values_length)

        decoder_output_train, decoder_state_train, decoder_len_train = name_decoder.train(
            embedded_cause=embedded_cause, cause_label=cause_label, cause_length=cause_length)
        cause_output_infer, cause_state_infer, cause_length_infer = name_decoder.infer(
            cause_encoder=cause_encoder)

        max_cause_length = params['max_cause_length']
        beam_width = params['beam_width']
        EOS = params['EOS']
        mask_for_cause = tf.sequence_mask(cause_length - 1, max_cause_length - 1, dtype=tf.float32)
        # loss = sequence_loss(logits=padded_train_output, targets=cause_label, weights=mask_for_cause, name='loss')
        padded_rnn_output = tf.pad(decoder_output_train.rnn_output,
                                   [[0, 0],
                                    [0, max_cause_length - 1 - tf.shape(decoder_output_train.rnn_output)[1]],
                                    [0, 0]],
                                   constant_values=0)
        loss = compute_loss(padded_rnn_output, cause_label, mask_for_cause)
        # loss = tf.Print(loss, [loss, 'loss'])

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
            optimizer = tf.train.AdamOptimizer()
            # grads_and_vars = optimizer.compute_gradients(loss)
            # clipped_gvs = [ele if ele[0] is None else (tf.clip_by_value(ele[0], -0.1, 0.1), ele[1]) for ele in
            #                grads_and_vars]
            # train_op = optimizer.apply_gradients(clipped_gvs, global_step=tf.train.get_global_step())
            train_op = optimizer.minimize(loss, global_step=tf.train.get_global_step())
            return tf.estimator.EstimatorSpec(mode=mode, loss=loss, train_op=train_op)

        # todo: now I have to leave the evaluation work be done outside the estimator
        eval_metric_ops = {
            'predicted_and_cause_ids': tf.contrib.metrics.streaming_concat(predicted_and_cause_ids),
        }
        return tf.estimator.EstimatorSpec(mode=mode, loss=loss, eval_metric_ops=eval_metric_ops)