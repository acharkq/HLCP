import pickle
import numpy as np
import json
from utilities.data_config import *


project_dir = Path(__file__).parent.parent
critical_bin_dir = project_dir / 'bin' / 'critical'
civil_bin_dir = project_dir / 'bin' / 'civil'


binDir2 = bin_dir
lixiangBin = os.path.join(binDir2, 'lixiang')



pathDict = {"dataDir": bin_dir,
            "dataDir2": bin_dir,
            "critical_bin_dir": critical_bin_dir,
            "binDir2": binDir2,
            "lixiangBin": lixiangBin,
            "civil_bin_dir": civil_bin_dir,
            'civilBaselines': os.path.join(civil_bin_dir, 'baselines'),
            "processedData": os.path.join(critical_bin_dir, "processedData"),
            "statistics": os.path.join(critical_bin_dir, "statistics"),
            "cuttedData": os.path.join(critical_bin_dir, "cuttedData"),
            "target_index": os.path.join(critical_bin_dir, "target_index"),
            'train_data': os.path.join(binDir2, 'train_data'),
            'cause_statistics': os.path.join(critical_bin_dir, 'cause_statistics'),
            'statistic_file': os.path.join(critical_bin_dir, 'statisticFile'),
            'dict_path': os.path.join(critical_bin_dir, 'dictionary.pkl'),
            'dict_path_civil': os.path.join(civil_bin_dir, 'word2vec', 'dictionary.pkl'),
            'model_dir': os.path.join(binDir2, 'model'),
            'standard_data_bin': os.path.join(binDir2, 'standard_data_bin'),
            'big_train': os.path.join(bin_dir, 'text_classification', 'big', 'train.txt'),
            'word2vec_bin': os.path.join(bin_dir, 'word2vec'),
            'word2vec_bin_civil': os.path.join(civil_bin_dir, 'word2vec'),
            'history': os.path.join(binDir2, 'history'),
            }


def getPath(target):
    try:
        return pathDict[target]
    except KeyError:
        raise ("Unexpected path")


word2vec_params = {'size': 100, 'min_count': 10, 'workers': 4, 'epochs': 10}

params = {'num_filters': 100, 'kernel_size': 5, 'num_units': 100, 'dropout_keep_prob': 0.5,
          'beam_width': 5, 'max_sequence_length': 500, 'warm_up_steps': 4000,
          # cnn_baseline settings
          'filter_sizes': [2, 3, 4, 5], 'num_filter': 25,
          # lstm baseline settings
          'num_layers': 2}
# max_cause_length is 5, but the length of the EOS mark should be considerer, too.

paramDict = {
    'sen_len': params['max_sequence_length'],  # max num of words in a sentence
    'min_fn': 10,  # minimum frequency of a word that won't be dropped
    'batch_size': 64,
    'num_epochs': 2000,
    'params': params,
    'word2vec_params': word2vec_params,
}


def getParam(target):
    try:
        return paramDict[target]
    except KeyError:
        raise ("Unexpected parameter")


def load_indexes(load_cause_dict=False, reverse=False, fix_dupin_duplicate=True, civil=False):
    '''load the indexes dicts of all labels, load cause_dict if True'''
    '''reverse to load the reverse dict'''

    def _hack(cause_dict):
        causes_dict['刑事']['妨害社会管理秩序罪']['走私、贩卖、运输、制造毒品罪'] = causes_dict['刑事']['妨害社会管理秩序罪'].pop('走私、贩卖、运输、制造毒品罪_父')
        causes_dict['刑事']['妨害社会管理秩序罪']['走私、贩卖、运输、制造毒品罪'].pop('走私、贩卖、运输、制造毒品罪')
        causes_dict['刑事']['妨害社会管理秩序罪']['走私、贩卖、运输、制造毒品罪']['走私、贩卖、运输、制造毒品罪2'] = {}
        return causes_dict

    if not civil:
        with open(getPath("target_index"), "r") as f:
            keyword2index = json.loads(f.readline())
            cause2index = json.loads(f.readline())
            law2index = json.loads(f.readline())
            result2index = json.loads(f.readline())
            causes_dict = json.loads(f.readline())
            causes_dict = fix_cause_dict(causes_dict, fix_dupin_duplicate=fix_dupin_duplicate)
            causes_dict = _hack(causes_dict)
        if fix_dupin_duplicate:
            cause2index['走私、贩卖、运输、制造毒品罪2'] = 10000
        if reverse:
            index2keyword = {item: label for label, item in keyword2index.items()}
            index2cause = {item: label for label, item in cause2index.items()}
            index2law = {item: label for label, item in law2index.items()}
            index2result = {item: label for label, item in result2index.items()}

            if load_cause_dict:
                return index2keyword, index2cause, index2law, index2result, causes_dict
            return index2keyword, index2cause, index2law, index2result
        else:
            if load_cause_dict:
                return keyword2index, cause2index, law2index, result2index, causes_dict
            return keyword2index, cause2index, law2index, result2index
    with open(os.path.join(getPath('civil_bin_dir'), 'cause2id'), 'rb') as f:
        cause2id = pickle.load(f)
    with open(os.path.join(getPath('civil_bin_dir'), 'final_sta'), 'rb') as f:
        tree = pickle.load(f)
    if not reverse:
        if load_cause_dict:
            return None, cause2id, None, None, tree
        return None, cause2id, None, None
    id2cause = {id: cause for cause, id in cause2id.items()}
    if load_cause_dict:
        return None, id2cause, None, None, tree
    return None, id2cause, None, None


class data_loader(object):
    def __init__(self, data_paths, mode, cause_projection=None, civil=True, layer_parent=(None, None)):
        if isinstance(data_paths, str):
            self.data_paths = [data_paths]
        self._mode = mode
        self._cause_projection = cause_projection
        self.gene = self.generator()
        self._layer_parent = layer_parent
        if civil:
            self.MAX_CAUSE_LENGTH2 = 4
        else:
            self.MAX_CAUSE_LENGTH2 = 3

    def __iter__(self):
        return self

    def __next__(self):
        '''mode: 0, load content only; 1, load cause labels only, mode: 2, load leaf labels only'''
        return next(self.gene)

    def _binarizer(self, causes):
        causes = [self._cause_projection[cause] for cause in causes]
        causes += (self.MAX_CAUSE_LENGTH2 - len(causes)) * [causes[0]]
        zeros = np.zeros([self.MAX_CAUSE_LENGTH2, len(self._cause_projection)])
        zeros[np.arange(self.MAX_CAUSE_LENGTH2), causes] = 1
        binarized = np.sum(zeros, 0, dtype=np.bool)
        binarized = binarized.astype(np.int32)
        return binarized

    def tiao_kuan_binarizer(self, tiao_list):
        zeros = np.zeros(shape=[484])
        zeros[tiao_list] = 1
        return zeros

    def generator(self):
        for data_path in self.data_paths:
            with open(data_path, 'r', encoding='utf-8') as f:
                while True:
                    line = f.readline()
                    if not line:
                        break
                    line = json.loads(line)
                    if self._mode == 0:
                        yield line['content']
                    elif self._mode == 1:
                        yield line['meta']['causes']
                    elif self._mode == 2:
                        yield line['meta']['causes'][-1]
                    elif self._mode == 3 and self._cause_projection:
                        yield self._binarizer(line['meta']['causes'])
                    elif self._mode == 4:
                        yield self.tiao_kuan_binarizer(line['meta']['laws'])
                    elif self._mode == 5:
                        yield list(map(self._cause_projection, line['meta']['causes']))
                    elif self._mode == 6:
                        yield int(line['meta']['causes'][0])
                    elif self._mode == 7:
                        yield self._cause_projection(line['meta']['causes'][-1])


NUM_OF_CAUSES_DATA = 333086


def fix_cause_dict(cause_dict, fix_dupin_duplicate=True):
    cause_dict['刑事'].pop('1979年至1997年间取消')
    cause_dict['刑事']['危害公共安全罪']['危险驾驶罪'].pop('协助组织卖淫罪')
    cause_dict['刑事']['危害公共安全罪']['非法持有、私藏枪支、弹药罪'].pop('容留他人吸毒罪')
    cause_dict['刑事']['侵犯公民人身权利民主权利罪']['非法拘禁罪'].pop('容留他人吸毒罪')
    cause_dict['刑事']['侵犯公民人身权利民主权利罪']['故意伤害罪'].pop('容留他人吸毒罪')
    cause_dict['刑事']['侵犯财产罪']['抢夺罪'].pop('容留他人吸毒罪')
    cause_dict['刑事']['侵犯财产罪']['故意毁坏财物罪'].pop('容留他人吸毒罪')
    cause_dict['刑事']['侵犯财产罪']['盗窃罪'].pop('容留他人吸毒罪')
    cause_dict['刑事']['侵犯公民人身权利民主权利罪']['绑架罪'].pop('寻衅滋事罪')
    cause_dict['刑事']['侵犯公民人身权利民主权利罪']['故意杀人罪'].pop('寻衅滋事罪')
    cause_dict['刑事']['侵犯公民人身权利民主权利罪']['非法拘禁罪'].pop('寻衅滋事罪')
    cause_dict['刑事']['侵犯公民人身权利民主权利罪']['故意伤害罪'].pop('寻衅滋事罪')
    cause_dict['刑事']['侵犯财产罪']['敲诈勒索罪'].pop('寻衅滋事罪')
    cause_dict['刑事']['侵犯财产罪']['故意毁坏财物罪'].pop('寻衅滋事罪')
    cause_dict['刑事']['危害公共安全罪']['非法持有、私藏枪支、弹药罪'].pop('开设赌场罪')
    cause_dict['刑事']['侵犯公民人身权利民主权利罪']['故意伤害罪'].pop('开设赌场罪')
    cause_dict['刑事']['侵犯财产罪']['敲诈勒索罪'].pop('开设赌场罪')
    cause_dict['刑事']['侵犯财产罪']['盗窃罪'].pop('开设赌场罪')
    cause_dict['刑事']['侵犯公民人身权利民主权利罪']['强奸罪'].pop('引诱、容留、介绍卖淫罪')
    cause_dict['刑事']['侵犯公民人身权利民主权利罪']['故意伤害罪'].pop('聚众斗殴罪')
    cause_dict['刑事']['侵犯财产罪']['盗窃罪'].pop('聚众斗殴罪')
    cause_dict['刑事']['侵犯公民人身权利民主权利罪']['故意伤害罪'].pop('赌博罪')
    cause_dict['刑事']['侵犯财产罪']['敲诈勒索罪'].pop('赌博罪')
    cause_dict['刑事']['侵犯财产罪']['盗窃罪'].pop('走私、贩卖、运输、制造毒品罪')
    cause_dict['刑事']['侵犯公民人身权利民主权利罪']['非法拘禁罪'].pop('走私、贩卖、运输、制造毒品罪')
    cause_dict['刑事']['危害公共安全罪']['交通肇事罪'].pop('非法持有毒品罪')
    cause_dict['刑事']['侵犯公民人身权利民主权利罪']['故意伤害罪'].pop('非法持有毒品罪')
    cause_dict['刑事']['侵犯财产罪']['盗窃罪'].pop('非法持有毒品罪')
    cause_dict['刑事']['侵犯财产罪']['盗窃罪'].pop('非法收购、运输、加工、出售国家重点保护植物、国家重点保护植物制品罪')
    cause_dict['刑事']['侵犯财产罪']['抢劫罪'].pop('00105013')
    cause_dict['刑事']['妨害社会管理秩序罪']['走私、贩卖、运输、制造毒品罪_父'] = cause_dict['刑事']['妨害社会管理秩序罪'].pop('走私、贩卖、运输、制造毒品罪')
    return cause_dict
