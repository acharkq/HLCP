import time, os, sys, json
import tensorflow as tf
from sys import stdout
from utilities.data_config import bin_dir
from utilities.DictionaryHeleper import DictionaryHelper
from utilities.EmbeddingHelper import EmbeddingHelper
from utilities.CauseHelper import CauseHelper


def cail_cause_projection(law=False):
    cail_data_bin = os.path.join(bin_dir, 'cail_0518')
    def _fix_cause_tree(cause_tree):
        cause_tree['刑事']['破坏社会主义市场经济秩序罪']['危害税收征管罪'].pop('虚开增值税专用发票、用于骗取出口退税、抵押税款发票罪')
        cause_tree['刑事']['破坏社会主义市场经济秩序罪']['危害税收征管罪']['虚开增值税专用发票、用于骗取出口退税、抵扣税款发票罪'] = {}
        return cause_tree

    from utilities.config_loader import cause_projection
    cause2index, cause_tree = cause_projection()
    with open(os.path.join(cail_data_bin, 'projection', 'projection.json'), 'r', encoding='utf8') as f:
        cause2count = json.loads(f.readline())
        law2count = json.loads(f.readline())
        cause2count = {cause + '罪': count for cause, count in cause2count.items()}
    cause_tree = _fix_cause_tree(cause_tree)
    if not law:
        return cause2count, cause_tree
    return cause2count, cause_tree, law2count


tf.logging.set_verbosity(tf.logging.INFO)


def build_estimator(model_fn, model_dir, params, warm_start_path=None):
    time_stamp = str(int(time.time()))
    model_dir = os.path.join(model_dir, time_stamp)
    session_config = tf.ConfigProto(
        allow_soft_placement=True,
        log_device_placement=False,
    )
    session_config.gpu_options.allow_growth = True
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)


    if not warm_start_path:
        estimator = tf.estimator.Estimator(model_fn=model_fn, params=params, model_dir=model_dir,
                                           config=tf.estimator.RunConfig(session_config=session_config,
                                                                         save_checkpoints_steps=1000,
                                                                         keep_checkpoint_max=2))
    else:
        import glob
        print(warm_start_path)
        datas = glob.glob(os.path.join(warm_start_path, '*'))
        datas.sort()
        print(datas)
        model_path = '.'.join(datas[-1].split('.')[:2])
        print(model_path)
        estimator = tf.estimator.Estimator(model_fn=model_fn, params=params,
                                           warm_start_from=model_path,
                                           model_dir=model_dir,
                                           config=tf.estimator.RunConfig(session_config=session_config,))
    return estimator


class train_method(object):
    parameters = {
        'num_filters': 256, 'kernel_size': 5, 'num_units': 256, 'dropout_keep_prob': 0.5,
        'beam_width': 5, 'sen_len': 500, 'batch_size': 128, 'epoch_num': 1000, 'device': '/gpu: 0',
    }

    def set_parameter(self, data_set):
        self.parameters['word_embeddings'] = data_set._embedding_helper.get_embedding()
        self.parameters['embedding_size'] = data_set._embedding_helper._dimension


class train_cail(train_method):
    _civil = False
    _data_bin = None
    _data_bin_name = None
    dataset = None
    estimator = None
    eval_listener = None
    cause_helper = None
    from utilities.data_config import critical_model_path

    def __init__(self, data_bin_name):
        self._civil = data_bin_name == 'civil'
        print('\n\n', data_bin_name, '\n\n')
        name2folder = {'cail': 'cail_0518', 'pku': 'pku', 'civil': 'civil'}
        name2folder = {name: os.path.join(bin_dir, folder) for name, folder in name2folder.items()}
        self._data_bin = os.path.join(bin_dir, name2folder[data_bin_name])
        self.dictionary = DictionaryHelper(bin_path=os.path.join(self._data_bin, 'word2vec'))
        self.embedding = EmbeddingHelper(bin_path=os.path.join(self._data_bin, 'word2vec'))

    def _load_causes(self, data_bin):
        with open(os.path.join(data_bin, 'projection', 'projection.json'), 'r', encoding='utf8') as f:
            cause2num = json.loads(f.readline())
            if not self._civil:
                cause2num = {cause + '罪': num for cause, num in cause2num.items()}
            self._cause2num = cause2num
            return set(self._cause2num)

    def train(self):
        # bin_dir = os.path.join(self._data_bin, 'train_data')
        bin_dir = self._data_bin
        tensors_to_log = {
            # "predicted_and_cause_ids_train": "model/predicted_and_cause_ids_train",
            "predicted_and_cause_ids": "model/predicted_and_cause_ids",
        }
        logging_hook = tf.train.LoggingTensorHook(
            tensors=tensors_to_log, every_n_iter=400)
        self.estimator.train(
            lambda: self.dataset.get_next([os.path.join(bin_dir, 'data_train.json'), ]),
            hooks=[logging_hook],
            saving_listeners=[
                self.eval_listener(self.parameters, self.estimator, self.dataset,
                                   os.path.join(bin_dir, 'data_valid.json'),
                                   os.path.join(bin_dir, 'data_test.json'),
                                   cause2num = self._cause2num)],)

    def analysis(self):
        bin_dir = os.path.join(self._data_bin, 'train_data')
        eval = self.eval_listener(self.parameters, self.estimator, self.dataset,
                                  os.path.join(bin_dir, 'data_valid.json'),
                                  os.path.join(bin_dir, 'data_test.json'),
                                  analysis=True,
                                  cause2num=self._cause2num)
        eval.__error_analysis__()

    def play(self):
        self.dataset._cause_helper.play()


class train_cail_hlcp(train_cail):

    def __init__(self, data_bin_name, warm_start_path=None):
        super(train_cail_hlcp, self).__init__(data_bin_name)
        from models.estimator_model import estimator_model_1
        from utilities.DataSet_v2 import DataSet_v2
        from train_model.EvalCheckPointSaverListener import EvalCheckPointSaverListener
        if not self._civil:
            cause2index, cause_tree = cail_cause_projection()
        else:
            from utilities.config_loader import cause_projection
            cause2index, cause_tree = cause_projection(civil=True)

        self.cause_helper = CauseHelper(self._load_causes(self._data_bin), cause_tree)
        self.dataset = DataSet_v2(self.parameters, self.dictionary, self.embedding, self.cause_helper)
        self.parameters = self.set_parameter(self.dataset)
        self.estimator = build_estimator(estimator_model_1, model_dir=self.critical_model_path, params=self.parameters,
                                         warm_start_path=warm_start_path)
        self.eval_listener = EvalCheckPointSaverListener

    def set_parameter(self, data_set):
        super(train_cail_hlcp, self).set_parameter(data_set)
        params = self.parameters
        params['hie'] = True
        params['num_layers'] = 2
        params['text_encoder'] = 'CNNEncoder'
        params['name_encoder'] = 'CauseEncoder'
        params['name_decoder'] = 'LSTM_Attention'
        params['EOS'] = data_set._cause_helper._EOS
        params['SOS'] = data_set._cause_helper._SOS
        params['num_causes'] = data_set._cause_helper.num_causes()
        params['vocab_size'] = data_set._dictionary_helper.vocab_size()
        params['max_cause_length'] = data_set._cause_helper._max_cause_length
        cause_table, cause_table_length = data_set.cause_table()
        params['cause_table'] = cause_table
        params['cause_table_length'] = cause_table_length
        cause_word_table, cause_word_table_length = data_set.cause_word_table()
        params['cause_word_table'] = cause_word_table
        params['cause_word_table_length'] = cause_word_table_length
        for key, value in params.items():
            if key not in {'word_embeddings', 'cause_table', 'cause_table_length', 'cause_word_table',
                           'cause_word_table_length'}:
                print(key, ': ', value)
        stdout.flush()
        return params



if __name__ == "__main__":
    if sys.argv[1] == 'cpu':
        os.environ['CUDA_VISIBLE_DEVICES'] = ''
        print('running on cpu')
        stdout.flush()
    else:
        try:
            os.environ['CUDA_VISIBLE_DEVICES'] = sys.argv[1]
            print('\n\nGPU num: ', sys.argv[1], end='\n\n')
            stdout.flush()
        except:
            os.environ['CUDA_VISIBLE_DEVICES'] = '1'

    cail = train_cail_hlcp('civil')
    cail.train()