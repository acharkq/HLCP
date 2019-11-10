import pickle
import numpy as np
import os
from utilities.thulac import Thulac
from utilities.SavedData import getPath

class HaoxiEmbedding(object):
    word_num = 0
    vec_len = 0
    word2id = None
    vec = None
    dictionary = None
    small_word_num = 0
    small_word2id = None
    small_vec = None

    def __init__(self, dictionary, civil=False, ):
        # print("begin to load word embedding")
        self.dictionary = dictionary
        if not civil:
            self.embedding_bin = getPath('word2vec_bin')
            self.small_word2id_path = os.path.join(self.embedding_bin, 'small_word2id.pkl')
            self.small_vec_nor_path = os.path.join(self.embedding_bin, 'small_vec_nor.npy')
        else:
            self.embedding_bin = getPath('word2vec_bin_civil')
            self.small_word2id_path = os.path.join(self.embedding_bin, 'small_word2id_civil.pkl')
            self.small_vec_nor_path = os.path.join(self.embedding_bin, 'small_vec_nor_civil.npy')
        if os.path.exists(self.small_word2id_path):
            with open(self.small_word2id_path, 'rb') as f:
                self.small_word_num, self.vec_len = pickle.load(f)
                self.small_word2id = pickle.load(f)
            self.small_vec = np.load(self.small_vec_nor_path)
        elif dictionary == None:
            with open(os.path.join(self.embedding_bin, 'word2id.pkl'), 'rb') as f:
                self.small_word_num, self.vec_len = pickle.load(f)
                self.small_word2id = pickle.load(f)
            self.small_vec = np.load(os.path.join(self.embedding_bin, 'vec_nor.npy'))
        else:
            print("loading haoxi embeddings")
            with open(os.path.join(self.embedding_bin, 'word2id.pkl'), "rb") as f:
                (self.word_num, self.vec_len) = pickle.load(f)
                self.word2id = pickle.load(f)
            self.vec = np.load(os.path.join(self.embedding_bin, 'vec_nor.npy'))
            print("load word embedding succeed")
            self.save_small_embedding()

    def transform_word(self, word, id=False):
        if not id:
            if self.word2id == None:
                try:
                    return self.small_vec[self.small_word2id[word]].astype(dtype=np.float32)
                except:
                    return self.small_vec[self.small_word2id['<UNK>']].astype(dtype=np.float32)
            try:
                return self.vec[self.word2id[word]].astype(dtype=np.float32)
            except:
                return self.vec[self.word2id['<UNK>']].astype(dtype=np.float32)
        else:
            if self.word2id == None:
                try:
                    return self.small_word2id[word]
                except:
                    return self.small_word2id['<UNK>']
            try:
                return self.word2id[word]
            except:
                return self.word2id['<UNK>']

    def _word2id(self, word):
        try:
            return self.word2id[word]
        except:
            return self.word2id['UNK']

    def transform_setence(self, sentence):
        sentence = sentence.split()
        return [self.transform_word(word) for word in sentence]

    def transform(self, sentences):
        '''transform sentences into indexs, accecpt sentence string list or generator, sentence should be tokenized by whitespace'''
        for sentence in sentences:
            word_index_list = self.transform_setence(sentence)
            yield word_index_list, len(word_index_list)

    def save_small_embedding(self):
        self.small_word2id = self.dictionary.vocabulary_._mapping
        self.small_vec = [self.vec[self._word2id("UNK")]]
        print("saving small vec data")
        small_id2word = {id: word for word, id in self.small_word2id.items()}
        for i in range(len(small_id2word)):
            if i == 0:
                continue
            self.small_vec.append(self.vec[self._word2id(small_id2word[i])])
        self.small_vec = np.array(self.small_vec, dtype=np.float32)
        np.save(self.small_vec_nor_path, self.small_vec)
        with open(self.small_word2id_path, 'wb') as f:
            pickle.dump([len(self.small_word2id), self.vec_len], f)
            pickle.dump(self.small_word2id, f)
        print('saving finished')

    def get_cause_embedding_inistializer(self, causes, vec=True, id=True, name=False):
        '''cause [[num_words] num_causes]'''
        cause_vecs = []
        cause_ids = []
        names = []
        fenci = Thulac()
        causes = [fenci.cut(cause) for cause in causes]
        pad_word = self.transform_word('<UNK>', id=True)
        for cause in causes:
            cause_vec = np.array([0 for _ in range(self.vec_len)], dtype=np.float32)
            cause_id = []
            cause = clear_cause(cause)
            if name:
                names.append(cause)
            for word in cause:
                if vec:
                    cause_vec += self.transform_word(word)
                else:

                    cause_id.append(self.transform_word(word, id=id))
            if vec:
                if np.linalg.norm(cause_vec) != 0:
                    cause_vec /= np.linalg.norm(cause_vec)
                cause_vecs.append(cause_vec)
            else:
                cause_ids.append(cause_id)
        if name:
            return names
        if vec:
            return np.array(cause_vecs)
        else:
            max_length = 0
            for cause_id in cause_ids:
                if len(cause_id) > max_length:
                    max_length = len(cause_id)
            max_length += 5
            cause_ids_prime = []
            cause_ids_length = []
            for cause_id in cause_ids:
                cause_ids_length.append(len(cause_id))
                cause_id += (max_length - len(cause_id)) * [pad_word]
                cause_ids_prime.append(cause_id)
            return np.array(cause_ids_prime), np.array(cause_ids_length)


def clear_cause(cause):
    # check
    if cause[-1][-1] == "罪" and len(cause[-1]) > 1:
        last_word = cause.pop()
        cause += [last_word[:-1] + '罪']
    if cause[-1][-2:] == '纠纷' and len(cause[-1]) > 2:
        last_word = cause.pop()
        cause += [last_word[:-1] + '纠纷']
    return cause