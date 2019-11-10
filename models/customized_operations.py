import tensorflow as tf

def masked_mean_pooling(input, input_length, sen_len):
    '''do masked mean pooling on the sequence_length dimension of a tensor with shape=[batch_size, sequence_length, feature]'''
    sequence_mask = tf.sequence_mask(input_length, maxlen=sen_len, dtype=tf.float32)
    sequence_mask = tf.expand_dims(sequence_mask, axis=2)
    tmp = tf.multiply(input, sequence_mask) / tf.reduce_sum(sequence_mask, axis=1)[:, tf.newaxis]
    pooled = tf.reduce_sum(tmp, axis=1)
    return pooled
