import json
import numpy as np
import tensorflow as tf
from utilities.EmbeddingHelper import EmbeddingHelper
from utilities.DictionaryHeleper import DictionaryHelper
from utilities.CauseHelper import CauseHelper, CauseHelperBase, CauseHelper2Baselines


class DataSetBase(object):
    _data_dict = {}
    _local = False
    _params = None
    _dictionary_helper = None
    _embedding_helper = None
    _cause_helper = None

    def __init__(self, params, dictionary_helper, embedding_helper, cause_helper):
        if isinstance(params, dict) \
                and isinstance(dictionary_helper, DictionaryHelper) \
                and isinstance(embedding_helper, EmbeddingHelper) \
                and isinstance(cause_helper, CauseHelperBase):
            self._dictionary_helper = dictionary_helper
            self._embedding_helper = embedding_helper
            self._cause_helper = cause_helper
            self._params = params

    def get_dataset(self, data_paths, training=True):
        if not str(data_paths) in self._data_dict:
            self._fill_data(data_paths)
        dataset = self._data_dict[str(data_paths)]
        if training:
            dataset = dataset.shuffle(1000, 1, reshuffle_each_iteration=True)
            dataset = dataset.repeat(self._params['epoch_num'])
        return dataset

    def get_next(self, data_paths, training=True):
        dataset = self.get_dataset(data_paths, training)
        dataset_iter = dataset.make_one_shot_iterator()
        features, labels = dataset_iter.get_next()
        return features, labels

    def _add_details(self, lines):
        raise NotImplementedError('Not implemented')

    def _fill_data(self, data_paths):

        if not isinstance(data_paths, list):
            data_paths = [data_paths]
        lines = []
        for data_path in data_paths:
            f = open(data_path, 'r')
            lines += f.readlines()
            f.close()

        feature_dict, label_dict = self._add_details(lines)
        dataset = tf.data.Dataset.from_tensor_slices((feature_dict, label_dict))
        dataset = dataset.batch(self._params['batch_size'])

        self._data_dict[str(data_paths)] = dataset


class DataSet_v2(DataSetBase):

    def __init__(self, params, dictionary_helper, embedding_helper, cause_helper):
        if isinstance(cause_helper, CauseHelper):
            self._cause_helper = cause_helper
        else:
            raise ('wrong type')
        super(DataSet_v2, self).__init__(params, dictionary_helper, embedding_helper, cause_helper)

    def _add_details(self, lines):
        def __add_details(line, feature_dict, label_dict):
            line = json.loads(line)
            content, length = self._dictionary_helper.transform_content(line['content'])
            if length <= 5:
                return
            causes, cause_length = self._cause_helper.transform(line['meta']['causes'])
            feature_dict['contents'].append(content)
            feature_dict['content_lengths'].append(length)

            label_dict['causes'].append(causes)
            label_dict['cause_lengths'].append(cause_length)

        feature_dict = {"contents": [], 'content_lengths': []}
        label_dict = {'causes': [], 'cause_lengths': []}
        for line in lines:
            __add_details(line, feature_dict, label_dict)
        return feature_dict, label_dict

    def cause_table(self):
        return self._cause_helper.cause_table()

    def cause_word_table(self):
        cause_words_list = self._cause_helper.cutted_cause_list()
        cause_words_list = [self._dictionary_helper.transform(cause_word) for cause_word in cause_words_list]
        max_cause_words = 0
        cause_words_length_list = []
        for cause_words in cause_words_list:
            cause_words_length_list.append(len(cause_words))
            max_cause_words = max(len(cause_words), max_cause_words)
        unk_id = self._dictionary_helper.word2id('<UNK>')
        cause_words_list = [cause_words + (max_cause_words - len(cause_words)) * [unk_id] for cause_words in
                            cause_words_list]
        return np.asarray(cause_words_list), np.asarray(cause_words_length_list)


class DataSet2Baselines(DataSetBase):

    '''
    this is the dataset class for multi filter sized cnn
    '''
    def __init__(self, params, dictionary_helper, embedding_helper, cause_helper):
        if isinstance(cause_helper, CauseHelper2Baselines):
            self._cause_helper = cause_helper
        else:
            raise ('wrong type')
        super(DataSet2Baselines, self).__init__(params, dictionary_helper, embedding_helper, cause_helper)

    def _add_details(self, lines):
        def __add_details(line, feature_dict, label_dict):
            line = json.loads(line)
            content, length = self._dictionary_helper.transform_content(line['content'])
            if length <= 5:
                return
            cause_vec = self._cause_helper.transform(line['meta']['cause'][-1])
            feature_dict['contents'].append(content)
            feature_dict['content_lengths'].append(length)
            label_dict['cause_vec'].append(cause_vec)

        feature_dict = {"contents": [], 'content_lengths': []}
        label_dict = {'cause_vec': [], }
        for line in lines:
            __add_details(line, feature_dict, label_dict)
        # label_dict = {label: np.asarray(value) for label, value in label_dict.items()}
        return feature_dict, label_dict
