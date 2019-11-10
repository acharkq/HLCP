import os
import sys
import time
import tensorflow as tf
from utilities.DataSet import DataSet
from utilities.SavedData import getParam, getPath
from EvalCheckPointSaverListener import EvalCheckpointSaverListener
from _model_functions import _match_model_fn, _match_model_fn_v6, _match_model_fn_v5

tf.logging.set_verbosity(tf.logging.INFO)


def match_estimator(params, model_dir, attention_decoder):
    time_stamp = str(int(time.time()))
    model_dir = os.path.join(model_dir, time_stamp)
    session_config = tf.ConfigProto(
        # device_count={'CPU': 1},
        allow_soft_placement=True,
        log_device_placement=False)
    session_config.gpu_options.allow_growth = True
    if not os.path.exists(model_dir):
        os.mkdir(model_dir)

    elif attention_decoder == 0:
        return tf.estimator.Estimator(model_fn=_match_model_fn, params=params, model_dir=model_dir,
                                      config=tf.estimator.RunConfig(session_config=session_config,
                                                                    save_checkpoints_steps=1000,
                                                                    keep_checkpoint_max=5))

    elif attention_decoder == 5:
        return tf.estimator.Estimator(model_fn=_match_model_fn_v5, params=params,
                                      model_dir=model_dir,
                                      config=tf.estimator.RunConfig(session_config=session_config,
                                                                    save_checkpoints_steps=1000,
                                                                    keep_checkpoint_max=5))
    elif attention_decoder == 6:
        return tf.estimator.Estimator(model_fn=_match_model_fn_v6, params=params,
                                      model_dir=model_dir,
                                      config=tf.estimator.RunConfig(session_config=session_config,
                                                                    save_checkpoints_steps=1000,
                                                                    keep_checkpoint_max=5))


def main(argv, paths):
    '''merged vocabulary'''
    os.environ['CUDA_VISIBLE_DEVICES'] = argv[1]
    tensors_to_log = {  # "predicted_cause_ids_train": "predicted_cause_ids_train",
        "predicted_and_cause_ids_train": "model/predicted_and_cause_ids_train",
        # "decoder/decoder_mask/penalty_bias": "model/decoder/decoder_mask/penalty_bias",
        "predicted_and_cause_ids": "model/predicted_and_cause_ids"}
    logging_hook = tf.train.LoggingTensorHook(
        tensors=tensors_to_log, every_n_iter=400)
    civil = True
    ds_train = DataSet(paths['train_path'], True, getParam('batch_size'), getParam('num_epochs'), cut_fn=10,
                       new_data=True, civil=civil)
    ds_eval = DataSet(paths['valid_path'], False, getParam('batch_size'), 1, cut_fn=10, new_data=True, civil=civil)

    ds_test = DataSet(paths['test_path'], False, getParam('batch_size'), 1, cut_fn=10, new_data=True, civil=civil)

    params = getParam('params')
    params['vocab_size'] = ds_train.size_of_vocabulary()
    params['EOS'] = ds_train.get_eos()
    params['SOS'] = ds_train.get_sos()
    params['max_cause_length'] = ds_train.get_max_cause_length()
    params['word2vec_model'] = ds_train.get_word2vec()
    cause_table, cause_table_length = ds_train.build_tensor_table()
    # for i, (array, length) in enumerate(zip(cause_table, cause_table_length)):
    #     print(str(i) + str(list(map(lambda i: str(ds_train.cause_project_name(i)) + '  ' + str(i), array))) + str(length))
    params['cause_table'] = cause_table
    params['cause_table_length'] = cause_table_length
    params['embedding_size'] = ds_train.get_embedding_size()
    # params['cause_embedding'] = ds_train.cause_embedding_initializer(vec=True)
    cause_id_table, cause_id_table_length = ds_train.cause_embedding_initializer(vec=False)

    params['cause_id_table'] = cause_id_table
    params['cause_id_table_length'] = cause_id_table_length
    params['encoder'] = 'cnn'
    try:
        params['hie'] = argv[3] == 'True'
    except:
        params['hie'] = True

    for key, value in params.items():
        if key != "cause_table" and key != 'word2vec_model' and key != 'cause_embedding':
            print(str(key) + ":" + str(value))
    estimator = match_estimator(params, getPath('model_dir'), attention_decoder=6)
    estimator.train(input_fn=lambda: ds_train.get_data_set(),
                    saving_listeners=[EvalCheckpointSaverListener(estimator, ds_eval, ds_test, params, civil)],
                    # hooks=[logging_hook],
                    )


if __name__ == '__main__':
    train_path = os.path.join(getPath('civil_bin_dir'), 'data_train.json')
    test_path = os.path.join(getPath('civil_bin_dir'), 'data_test.json')
    valid_path = os.path.join(getPath('civil_bin_dir'), 'data_valid.json')
    paths = {'train_path': train_path, 'test_path': test_path, 'valid_path': valid_path}
    main(sys.argv, paths)
