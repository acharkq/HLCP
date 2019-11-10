import numpy as np
from utilities.thulac import Thulac
from utilities.internal_tools import tree_find_relevant_nodes, tree_find_kids


class CauseHelperBase(object):
    _leaves = set()
    _civil = False
    _cause2index = None
    _cause_list = None
    _cause_tree = None
    _cause2num = None

    def __init__(self, leaves, cause_tree):
        if '民事' in cause_tree:
            self._civil = True
            self._max_cause_length = 6  # this length include ths sos symbol and eos symbol
        elif '刑事' in cause_tree:
            self._civil = False
            self._max_cause_length = 5  # this length include ths sos symbol and eos symbol
        else:
            raise ('既不是刑事也不是民事')
        self._leaves = leaves
        self._cause_tree = cause_tree

    def leaves(self, index=True):
        if not index:
            return self._leaves
        return [self._cause2index[leaf] for leaf in self._leaves]

    def get_index2cause(self):
        index2cause = {index: cause for cause, index in self._cause2index.items()}
        return index2cause

    def get_cause2index(self):
        return self._cause2index

    def num_causes(self):
        return len(self._cause2index)

    def transform(self, causes):
        raise NotImplementedError()

    def play(self):
        index2cause = {index: cause for cause, index in self._cause2index.items()}
        while True:
            index = input('输入一个index, 输入q退出\n')
            if index == 'q':
                exit()
            try:
                index = int(index)
                cause = index2cause[index]
                print(cause)
            except:
                print('非法index， 请重新输入')


class CauseHelper(CauseHelperBase):
    _EOS = -1
    _SOS = -1
    _max_cause_length = -1
    _leaves = []
    _civil = False
    _cause2index = None
    _cause_list = None
    _cause_tree = None

    def __init__(self, leaves, cause_tree):
        super(CauseHelper, self).__init__(leaves, cause_tree)
        self._cause_list = list(tree_find_relevant_nodes(cause_tree, leaves))
        self._cause_list.sort()
        self._cause_list.append('EOS')
        self._cause2index = {cause: i for i, cause in enumerate(self._cause_list)}
        self._SOS = self._cause2index['民事'] if self._civil else self._cause2index['刑事']
        self._EOS = len(self._cause2index) - 1

    def transform(self, causes):
        # print(self._cause2index)
        # causes = [self._cause2index[cause] for cause in causes]
        length = len(causes)
        causes += (self._max_cause_length - length) * [self._EOS]
        return causes, length

    def cause_table(self):
        table = []
        table_length = []
        max_kids_num = 0
        for cause in self._cause_list:
            if cause == 'EOS':
                table.append([self._EOS])
                table_length.append(1)
                continue
            kids = tree_find_kids(self._cause_tree, cause)
            kids = [kid for kid in kids if kid in self._cause2index]
            kids = kids if kids else ['EOS']
            kids = [self._cause2index[kid] for kid in kids]
            if kids is None:
                raise ('One key out of the tree')
            table.append(kids)
            # todo: EOS符号在之前的模型中被算作序列生成的一部分
            table_length.append(len(kids))
            max_kids_num = max(max_kids_num, len(kids))
        for i in range(len(table)):
            table[i] += (max_kids_num - len(table[i])) * [table[i][-1]]
        return np.asarray(table), np.asarray(table_length)

    def cause_table_non_array(self):
        table = []
        for cause in self._cause_list:
            if cause == 'EOS':
                table.append([])
                continue
            kids = tree_find_kids(self._cause_tree, cause)
            kids = [kid for kid in kids if kid in self._cause2index]
            kids = [self._cause2index[kid] for kid in kids]
            table.append(kids)
        return table

    def cutted_cause_list(self):
        fenci = Thulac()
        cause_words_list = [fenci.cut(cause) for cause in self._cause_list]
        fenci.clear()
        return cause_words_list


class CauseHelper2Baselines(CauseHelperBase):
    _num_causes = -1

    def __init__(self, leaves, cause_tree):
        super(CauseHelper2Baselines, self).__init__(leaves, cause_tree)
        self._cause_list = list(self._leaves)
        self._cause_list.sort()
        self._cause2index = {cause: i for i, cause in enumerate(self._cause_list)}
        self._num_causes = len(self._cause_list)

    def transform(self, cause):
        index = self._cause2index[cause]
        # zeros = np.zeros(shape=[self._num_causes])
        # zeros[index] = 1
        return index