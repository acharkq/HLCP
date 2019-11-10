import os, pickle
from utilities.internal_tools import re_index


class DictionaryHelper(object):
    ''''''
    _vocab_size = 0
    _min_count = 0
    _sen_len = 500
    _word2id = None
    _word2count = None

    def __init__(self, min_count=0, sen_len=500, word2id=None, word2count=None, bin_path=None):
        self._sen_len = sen_len
        if word2id is not None and word2count is not None:
            self.__hidden__init__(min_count, word2id, word2count)
        elif bin_path is not None:
            self.restore(bin_path)

    def vocab_size(self):
        return self._vocab_size

    def fit(self, text_generator):
        self._word2id = {}
        self._word2count = {}
        self._fit(text_generator)

    def get_dictionary(self):
        return self._word2id

    def word2id(self, word):
        if not isinstance(word, str):
            raise ('word must be a string')
        try:
            return self._word2id[word]
        except KeyError:
            return self._word2id['<UNK>']

    def transform(self, sentence):
        def _escape_zhong(sentence):
            sentence_prime = []
            for sen in sentence:
                sentence_prime += sen
            return sentence_prime

        def _trans_word(word):
            if word in self._word2id:
                return self._word2id[word]
            else:
                return self._word2id['<UNK>']

        sentence = _escape_zhong(sentence)
        sentence = [_trans_word(word) for word in sentence]
        return sentence

    def transform_content(self, sentence):
        def _trans_word(word):
            if word in self._word2id:
                return self._word2id[word]
            else:
                return self._word2id['<UNK>']
        if not isinstance(sentence[0], str):
            sentence_prime = []
            for sub_sen in sentence:
                sentence_prime += sub_sen
            sentence = sentence_prime
        sen_len = len(sentence)
        sentence = [_trans_word(word) for word in sentence][:self._sen_len]
        sentence += (self._sen_len - sen_len) * [self._word2id['<padding>']]
        return sentence, sen_len

    def _fit(self, text_generator):
        '''text generator generate whitespace-delimited text'''

        def _add(word, count=1):
            if not len(word) >= 20:
                self._word2id[word] = self._vocab_size
                self._vocab_size += 1
                self._word2count[word] = count

        _add('<UNK>', 100000)
        _add('<padding>', 100000)
        for text in text_generator:
            if not isinstance(text, list):
                text = text.split()
            for word in text:
                if not len(word) >= 20:
                    if word in self._word2id:
                        self._word2count[word] += 1
                    else:
                        _add(word)

    def save(self, bin_path):
        with open(os.path.join(bin_path, 'dictionary.pkl'), 'wb') as f:
            pickle.dump(self._min_count, f)
            pickle.dump(self._word2id, f)
            pickle.dump(self._word2count, f)

    def restore(self, bin_path):
        with open(os.path.join(bin_path, 'dictionary.pkl'), 'rb') as f:
            self._min_count = pickle.load(f)
            self._word2id = pickle.load(f)
            self._word2count = pickle.load(f)
            self._vocab_size = len(self._word2count)

    def __hidden__init__(self, min_count, word2id, word2count):
        __check__(word2id, word2count)
        self._word2id = word2id
        self._word2count = word2count
        self._vocab_size = len(self._word2id)
        self._trim(min_count)

    def _trim(self, min_count):
        '''trim will distrupt the '''
        if min_count <= self._min_count:
            return
        self._min_count = min_count
        self._word2count = {word: count for word, count in self._word2count.items() if count >= self._min_count}
        self._word2id = {word: id for word, id in self._word2id.items() if word in self._word2count}
        self._word2id = re_index(self._word2id, self._vocab_size)
        self._vocab_size = len(self._word2id)

    def intersection_update(self, word2id):
        words1 = set(self._word2id)
        words2 = set(word2id)
        # keep the special character
        words2.add('<UNK>')
        words2.add('<padding>')
        words = words1.intersection(words2)
        self._word2count = {word: count for word, count in self._word2count.items() if word in words}
        self._word2id = {word: id for word, id in self._word2id.items() if word in words}
        self._word2id = re_index(self._word2id, self._vocab_size)
        self._vocab_size = len(self._word2id)


def __check__(word2id, word2count):
    errors = []
    if len(word2id) != len(word2count):
        errors.append('词数不一样')
    words1 = set(word2id)
    words2 = set(word2count)
    if len(words1.intersection(words2)) != len(words1):
        errors.append('单词不对应')
    if '<UNK>' not in word2id:
        errors.append('缺少空字符')
    if '<padding>' not in word2id:
        errors.append('缺少padding字符')
    if errors:
        print('这么多问题啊')
        for error in errors:
            print(error)
        exit(1)
