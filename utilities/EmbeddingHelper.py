import pickle, os
import numpy as np


class EmbeddingHelper(object):
    _dimension = 0
    _word_num = 0
    _word2id = None
    _id2embedding = None

    def __init__(self, word2id=None, id2embedding=None, bin_path=None):
        '''id2embedding 必须是 numpy array'''
        if word2id is not None and id2embedding is not None:
            self.__hidden__init__(word2id, id2embedding)
        elif bin_path is not None:
            self.restore(bin_path)

    def __hidden__init__(self, word2id, id2embedding):
        __check__(word2id, id2embedding)
        self._word2id = word2id
        self._id2embedding = np.asarray(id2embedding, dtype=np.float32)
        self._word_num = len(word2id)
        self._dimension = len(id2embedding[0])

    def save(self, bin_path):
        with open(os.path.join(bin_path, 'word2id.pkl'), 'wb') as f:
            pickle.dump(self._word2id, f)
        np.save(os.path.join(bin_path, 'id2embedding.npy'), self._id2embedding)

    def restore(self, bin_path):
        with open(os.path.join(bin_path, 'word2id.pkl'), 'rb') as f:
            word2id = pickle.load(f)
        id2embedding = np.load(os.path.join(bin_path, 'id2embedding.npy'))
        self.__hidden__init__(word2id, id2embedding)

    def get_embedding(self):
        return self._id2embedding

def __check__(word2id, id2embedding):
    errors = []
    if len(word2id) != len(id2embedding):
        errors.append('你在逗我?词数不一样的')
    if '<UNK>' not in word2id:
        errors.append('缺少空字符')
    if '<padding>' not in word2id:
        errors.append('缺少padding字符')
    if not isinstance(id2embedding, np.ndarray):
        errors.append('id2embedding 仅接受np array 实例')
    if errors:
        print('这么多问题啊')
        for error in errors:
            print(error)
        exit(1)
