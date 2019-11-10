import os
import json
import tensorflow as tf
import numpy as np
from utilities.SavedData import getPath, getParam, data_loader, load_indexes
from utilities.DataHelpers import get_valid_cause
from tensorflow.contrib import learn

from utilities.Word_Embedding_Helper import HaoxiEmbedding


# import tensorflow.contrib.eager as tfe
# tfe.enable_eager_execution()

def count_sen_length(sentence):
    sentence = sentence[::-1]
    sen_len = len(sentence)
    for word in sentence:
        if word == 0:
            sen_len -= 1
        else:
            break
    return sen_len


class DataSet(object):
    MAX_CAUSE_LENGTH = 5  # '刑事'+ ~ +'EOS'
    SEN_LEN = getParam('sen_len')
    MIN_SEN_LEN = 10
    num_threads = 5
    dupin_index = 5
    dupin_duplicate_index = 10000

    # max_kids_num = 34  # the real num is 33, make it larger for redudant

    '''I removed the 刑法 cause at the beginning of every causes'''

    def __init__(self, data_path, is_train, batch_size, num_epochs, cut_fn=200, word2vec=False, new_data=False,
                 civil=False):
        if civil:
            self.SOSINDEX = 0
            self.MAX_CAUSE_LENGTH = 6
        else:
            self.SOSINDEX = 243
            self.MAX_CAUSE_LENGTH = 5

        self.is_train = is_train
        self.data_path = data_path
        self.batch_size = batch_size
        self.num_epochs = num_epochs
        self.cut_fn = cut_fn
        self.new_data = new_data
        self.civil = civil
        causes = get_valid_cause(self.cut_fn, new_data, civil)
        self.build_cause_projection(causes)
        self.word2vec = word2vec
        if word2vec:
            exit("word2vec self trained bin not available now")
        else:
            self.load_dictionary()
            data_path += '_dict'
        if cut_fn > 0:
            self.tf_files = [data_path + '%d' % i + '_%d.tfrecords' % cut_fn for i in range(self.num_threads)]
            # self.tf_file = data_path + '_%d.tfrecords' % cut_fn
        else:
            self.tf_files = None

    def get_word2vec(self):
        if self.word2vec:
            return self.weh.get_word2vec()
        else:
            return self.word_embedding

    def get_embedding_size(self):
        return self.haoxi_embedding.vec_len

    def load_dictionary(self):
        if not self.civil:
            if os.path.exists(getPath('dict_path')):
                self.dictionary = learn.preprocessing.VocabularyProcessor.restore(getPath('dict_path'))
            else:
                text_gene = data_loader(self.data_path, 0)
                self.dictionary = learn.preprocessing.VocabularyProcessor(max_document_length=self.SEN_LEN,
                                                                          min_frequency=10)
                self.dictionary.fit(text_gene)
                self.dictionary.save(getPath('dict_path'))
        else:
            print(os.path.exists(getPath('dict_path_civil')))
            if os.path.exists(getPath('dict_path_civil')):
                self.dictionary = learn.preprocessing.VocabularyProcessor.restore(getPath('dict_path_civil'))
            else:
                text_gene = data_loader(self.data_path, 0)
                self.dictionary = learn.preprocessing.VocabularyProcessor(max_document_length=self.SEN_LEN,
                                                                          min_frequency=10)
                self.dictionary.fit(text_gene)
                self.dictionary.save(getPath('dict_path_civil'))
        self.haoxi_embedding = HaoxiEmbedding(self.dictionary, self.civil)
        self.word_embedding = self.haoxi_embedding.small_vec

    def build_cause_projection(self, causes):
        self.causes_set = causes
        self.causes_list = list(causes)
        self.causes_list.sort()
        # to project the cause into a continuous range
        self.causes_projection = {cause: i for i, cause in enumerate(self.causes_list)}
        # add the start of sequence which is 刑事 and end of sequence
        self.SOS = self.causes_projection[self.SOSINDEX]
        self.EOS = len(self.causes_projection)

        # print(len(self.causes_projection))
        self.num_causes = self.EOS + 1
        self.causes_projection['EOS'] = self.EOS
        self.causes_projection_reversed = {i: cause for cause, i in self.causes_projection.items()}

    def cause_project_name(self, cause_num, leaf=False, final=True):
        if leaf:
            leaf_index = self.get_leaf_index(leaf_projection=True)
            index_leaf = {index: leaf for leaf, index in leaf_index.items()}
            cause_num = index_leaf[cause_num]
        if final:
            cause_num = self.causes_projection_reversed[cause_num]
        if cause_num == 'EOS':
            return 'EOS'
        index2keyword, index2cause, index2law, index2result = load_indexes(reverse=True, civil=self.civil)
        cause_name = index2cause[cause_num]
        return cause_name

    def get_cause_name_list(self):
        cause_names = []
        for i in self.causes_projection_reversed:
            cause_names.append(self.cause_project_name(i))
        return cause_names

    def build_tensor_table(self):
        keyword2index, cause2index, law2index, result2index, cause_dict = load_indexes(load_cause_dict=True,
                                                                                       civil=self.civil)
        index2keyword, index2cause, index2law, index2result = load_indexes(reverse=True, civil=self.civil)
        self.cause_dict = cause_dict
        table = []
        length_table = []
        max_kids_num = 0
        for i, cause in enumerate(self.causes_list):
            duiying = self.tree_find_kids(cause_dict, index2cause[cause])
            duiying_prime = []
            if duiying:
                for cause_sub in duiying:
                    try:
                        duiying_prime.append(self.causes_projection[cause2index[cause_sub]])
                    except KeyError:
                        pass
            table.append(duiying_prime if len(duiying_prime) else [self.EOS])
            length_table.append(max(len(duiying_prime), 1))
            if len(duiying_prime) > max_kids_num:
                max_kids_num = len(duiying_prime)
            # todo: 此处尚有不确定之处, 即是否应将EOS算进attention, 我感觉是不用的
            # todo: 2018-4-28 edit, 我真的是吃了shit了, 前一行这个todo我在写这里的时候都注意到了,他妈的老早老早就感觉这里会出问题,结果出了问题之后硬是没想到是这里的问题,debug了快一天才找到这里
        # exit(0)
        max_kids_num += 1
        table.append([self.EOS] * max_kids_num)
        length_table.append(1)
        table_prime = []
        for duiying in table:
            duiying += (max_kids_num - len(duiying)) * [duiying[-1]]
            table_prime.append(duiying)
        # for the eos line
        return table_prime, length_table

    def get_leaf_index(self, name=False, leaf_projection=False):
        keyword2index, cause2index, law2index, result2index, cause_dict = load_indexes(load_cause_dict=True,
                                                                                       civil=self.civil)

        def get_leaf(tree, box):
            for node in tree:
                if not tree[node]:
                    box.append(node)
                else:
                    get_leaf(tree[node], box)

        box = []
        get_leaf(cause_dict, box)
        box = [leaf for leaf in box if cause2index[leaf] in self.causes_set]
        if name:
            return box
        valid_leaves = [self.causes_projection[cause2index[leaf]] for leaf in box if
                        cause2index[leaf] in self.causes_set]
        valid_leaves.sort()
        if not leaf_projection:
            return valid_leaves
        return {leaf: i for i, leaf in enumerate(valid_leaves)}

    def tree_find_kids(self, tree, key):
        # todo, edit the data and seperate
        if key in tree:
            return tree[key]
        for node in tree:
            if tree[node]:
                if self.tree_find_kids(tree[node], key):
                    return self.tree_find_kids(tree[node], key)
        return None

    def tree_find(self, tree, key, box):
        if key in tree:
            box.append(key)
            return True
        have = False
        for node in tree:
            if tree[node]:
                if self.tree_find(tree[node], key, box):
                    box.append(node)
                    have = True
        return have

    def size_of_vocabulary(self):
        if self.word2vec:
            return len(self.weh.w2v.wv.vocab)
        else:
            return len(self.dictionary.vocabulary_)

    def get_eos(self):
        return self.EOS

    def get_sos(self):
        return self.SOS

    def get_max_cause_length(self):
        return self.MAX_CAUSE_LENGTH

    def get_data_set(self):
        if self.tf_files:
            if not os.path.exists(self.tf_files[0]):
                self.convert_to()
            ds = self._tf_data_set()
        else:
            ds = self._text_data_set()
        if self.is_train:
            ds = ds.shuffle(1000)
        ds = ds.prefetch(1000)
        ds = ds.batch(self.batch_size)
        ds = ds.repeat(self.num_epochs)
        it = ds.make_one_shot_iterator()
        features, labels = it.get_next()
        return features, labels

    def _text_data_set(self):
        ds = tf.data.TextLineDataset([self.data_path])
        ds = ds.map(self._wrapped_parse_line)
        ds = ds.filter(self._filter)
        return ds

    def _tf_data_set(self):
        ds = tf.data.TFRecordDataset(self.tf_files, num_parallel_reads=self.num_threads)
        ds = ds.map(self._parse_function, num_parallel_calls=self.num_threads)
        return ds

    def _cause_process(self, causes):
        """
        edited 2018-4-14
        add feature: deal with 走私、贩卖、运输、制造毒品罪 duplicate
        edited 2018-4-16
        add start of sequence token "刑事" at the begginning of every sentence
        edited 2018-5-10
        fix the issue about dupin index  while civil training
        """
        projected = [self.SOS]
        for i in range(len(causes)):
            if i != 0 and causes[i] == self.dupin_index and causes[i - 1] == self.dupin_index and not self.civil:
                projected.append(self.causes_projection[self.dupin_duplicate_index])
            else:
                projected.append(self.causes_projection[causes[i]])
        return projected + [self.EOS] * (self.MAX_CAUSE_LENGTH - len(projected))

    def sentence_tranformer(self, content_gene):
        def _dictioanry_transform_wrapper(content_gene):
            content_vec_gene = self.dictionary.transform(content_gene)
            for content_vec in content_vec_gene:
                yield content_vec, count_sen_length(content_vec)

        if self.word2vec:
            return self.weh.transform(content_gene)
        else:
            return _dictioanry_transform_wrapper(content_gene)

    def eval_cause_gene(self):
        content_gene = data_loader(self.data_path, 0)
        cause_gene = data_loader(self.data_path, 1)
        content_vec_gene = self.sentence_tranformer(content_gene)
        for content_vec in content_vec_gene:
            next_cause = self._abandon(next(cause_gene))
            if next_cause == None or np.count_nonzero(content_vec) < self.MIN_SEN_LEN:
                continue
            # todo: evaluate multi causes
            yield self._cause_process(next_cause)

    def convert_to(self):
        """Converts a dataset to tfrecords."""
        content_gene = data_loader(self.data_path, 0)
        cause_gene = data_loader(self.data_path, 1)
        content_vec_and_length_gene = self.sentence_tranformer(content_gene)
        print('Writing to tf_files')
        self.count = 0
        writers = [tf.python_io.TFRecordWriter(tf_file) for tf_file in self.tf_files]
        # with tf.python_io.TFRecordWriter(self.tf_file) as writer:
        for content_vec, content_length in content_vec_and_length_gene:
            next_cause = next(cause_gene)
            next_cause = self._abandon(next_cause)
            if next_cause == None or content_length < self.MIN_SEN_LEN:
                continue
            self.count += 1
            cause_length = len(
                next_cause) + 1  # one for eos, we don't count sos, cause it will not be calculated while computing loss
            next_cause = self._cause_process(next_cause)
            example = tf.train.Example(
                features=tf.train.Features(
                    feature={
                        'content': tf.train.Feature(int64_list=tf.train.Int64List(value=content_vec)),
                        'content_length': tf.train.Feature(
                            int64_list=tf.train.Int64List(value=[content_length])),
                        'cause_label': tf.train.Feature(int64_list=tf.train.Int64List(value=next_cause)),
                        'cause_length': tf.train.Feature(int64_list=tf.train.Int64List(value=[cause_length]))
                    }))

            writers[self.count % self.num_threads].write(example.SerializeToString())
        for writer in writers:
            writer.close()
        print('count ' + str(self.count))

    def _abandon(self, causes):
        '''abandon causes which are not common'''
        for cause in causes:
            if not cause in self.causes_set:
                return None
        return causes

    def _parse_line(self, line):
        line = json.loads(line)
        content = next(self.sentence_tranformer([line['content']]))
        content_length = np.count_nonzero(content)
        causes = line['meta']['causes']
        causes.append(self.EOS)
        causes = [self.SOS] + causes
        causes_length = len(causes)
        causes += (self.MAX_CAUSE_LENGTH - causes_length) * [self.EOS]
        return content, content_length, causes, causes_length

    def _wrapped_parse_line(self, line):
        content, content_length, causes, causes_length = tf.py_func(self._parse_line, [line], [tf.int64] * 4)
        features = {
            'content': content,
            'content_length': content_length,
        }
        labels = {
            'cause_label': causes,
            'cause_length': causes_length
        }
        if self.is_train:
            return features, labels
        else:
            return features

    def _out_range(self, total, b):
        for a in b:
            if a not in total:
                return True
        return False

    def _filter(self, features, labels=None):
        cond1 = tf.cond(features['content_length'] > self.MIN_SEN_LEN, true_fn=lambda: True, false_fn=lambda: False)
        return cond1

    def _parse_function(self, example_proto):
        info = {
            'content': tf.FixedLenFeature([self.SEN_LEN], tf.int64),
            'content_length': tf.FixedLenFeature((), tf.int64),
            'cause_label': tf.FixedLenFeature([self.MAX_CAUSE_LENGTH], tf.int64),
            'cause_length': tf.FixedLenFeature((), tf.int64)
        }
        parsed_info = tf.parse_single_example(example_proto, info)
        parsed_features = {'content': parsed_info['content'], 'content_length': parsed_info['content_length']}
        parsed_labels = {'cause_label': parsed_info['cause_label'], 'cause_length': parsed_info['cause_length']}
        return (parsed_features, parsed_labels)

    def cause_embedding_initializer(self, vec, id=True, name=False):
        cause_list = self.get_cause_name_list()
        return self.haoxi_embedding.get_cause_embedding_inistializer(cause_list, vec, id, name)