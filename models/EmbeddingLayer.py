import tensorflow as tf
import numpy as np
from tensorflow.python.ops import gen_array_ops


class EmbeddingLayer(object):
    _dimension_size = 0
    _embedding_matrix = None

    def __init__(self, embeddings):
        '''
        load embeddings from numpy array
        '''
        if not isinstance(embeddings, np.ndarray):
            raise ('embeddings must be numpy array')
        self._dimension_size = embeddings.shape[-1]
        self._embedding_matrix = tf.get_variable(initializer=embeddings, trainable=True, name='word_embedding',
                                                 dtype=tf.float32)

    def __call__(self, input):
        '''
        transform word ids into embeddings
        :param input: shape = [-1, sen_len]
        :return: shape = [-1, sen_len, embedding_dimension]
        '''
        return gen_array_ops.gather_v2(self._embedding_matrix, input, axis=0)
