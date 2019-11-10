import time, os
import numpy as np
import shutil, glob
from sys import stdout
import tensorflow as tf
from utilities.DataSet_v2 import DataSetBase, DataSet_v2
from collections import namedtuple
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix


class Result(namedtuple('Result', ('step', 'accuracy', 'precision', 'recall', 'fscore'))):
    pass


class EvalCheckPointSaverListenerBase(tf.train.CheckpointSaverListener):
    _results = []
    _params = {}
    _leaf_indexes = []

    _eval_step = 0
    _model_dir = None
    _estimator = None
    _dataset = None
    _valid_paths = None
    _test_paths = None
    _analysis = False

    early_stop_steps = 20

    max_validation_result = Result(0, 0, 0, 0, 0)
    test_result = Result(0, 0, 0, 0, 0)

    def __init__(self, params, estimator, dataset, valid_paths, test_paths=None, analysis=False, cause2num=None):
        '''
        由于目前采取的是钟皓曦的策略，所以我就不用early stop 什么的了
        '''

        if not isinstance(params, dict) or not isinstance(estimator, tf.estimator.Estimator) or not isinstance(dataset,
                                                                                                               DataSetBase):
            raise ('Not the right type input')
        self._analysis = analysis
        self._cause2num = cause2num
        self._params = params
        self._estimator = estimator
        self._dataset = dataset
        self._model_dir = self._estimator.model_dir
        self._optimal_model_dir = os.path.join(self._model_dir, 'optimal_model')
        self._leaf_indexes = self._dataset._cause_helper.leaves(index=True)
        if not os.path.exists(self._optimal_model_dir):
            os.mkdir(self._optimal_model_dir)
        self._valid_paths = valid_paths if isinstance(valid_paths, list) else [valid_paths]
        self._test_paths = test_paths if isinstance(test_paths, list) else [test_paths]

    def after_save(self, session, global_step_value, ):
        if not self._analysis:
            print('evaluation_started')
            self.__evaluate__()
            self.__early_stop__()
            print('highest validation score' + str(self.max_validation_result))
            print('test score' + str(self.test_result))
            print('evaluation_ended')
        else:
            print('analysis_started')
            self.__error_analysis__()
            print('analysis_ended')

        stdout.flush()

    def process_evaluate_results(self, evaluate_results):
        raise NotImplementedError()

    def __evaluate__(self):
        self._eval_step += 1
        evaluate_results_valid = self._estimator.evaluate(
            input_fn=lambda: self._dataset.get_next(self._valid_paths, training=False))
        target_list, predicts = self.process_evaluate_results(evaluate_results_valid)
        confusion = confusion_matrix(target_list, predicts, self._leaf_indexes)
        print(time.asctime(time.localtime(time.time())))
        accuracy, precisions, recalls, fscores = _get_scores(target_list, predicts, self._leaf_indexes)
        print('accuracy score %f' % accuracy)
        print('precsion  ' + str(precisions) + ' recall ' + str(recalls) + ' fscores ' + str(fscores))
        print('macro average precision recall f1-score: %f %f %f' %
              (precisions, recalls, fscores))
        print_confusion_matrix(confusion, self._leaf_indexes,
                               lambda x: self._dataset._cause_helper.get_index2cause()[x], rate=fscores, )
        self._results.append(Result(self._eval_step, accuracy, precisions, recalls, fscores))
        print('\n\n')

        # print('and the test score' + str(self.max_test_result))

    def __early_stop__(self):

        if self._results[-1].fscore >= self.max_validation_result.fscore:
            self.max_validation_result = self._results[-1]
            if self._eval_step < self.early_stop_steps:
                return
            latest_checkpoint = self._estimator.latest_checkpoint()
            shutil.rmtree(self._optimal_model_dir)
            os.mkdir(self._optimal_model_dir)
            for file in glob.glob(latest_checkpoint + '*'):
                shutil.copy(file, self._optimal_model_dir)
            evaluate_results_test = self._estimator.evaluate(
                input_fn=lambda: self._dataset.get_next(self._test_paths, training=False))
            target_list, predicts = self.process_evaluate_results(evaluate_results_test)
            accuracy, precisions, recalls, fscores = _get_scores(target_list, predicts, self._leaf_indexes)
            self.test_result = Result(self._eval_step, accuracy, precisions, recalls, fscores)

            from utilities.score_by_rank import evaluate_on_former_layer
            index2cause = self._dataset._cause_helper.get_index2cause()
            cause_tree = self._dataset._cause_helper._cause_tree
            evaluate_on_former_layer(lambda x: index2cause[x], cause_tree, predicts, target_list, self._leaf_indexes,
                                     self._cause2num)

        if self._eval_step < self.early_stop_steps:
            return
        # early stop
        for i in range(max(0, len(self._results) - self.early_stop_steps), max(len(self._results) - 1, 0)):
            if self._results[i].fscore < self._results[i + 1].fscore:
                return

        print("max validation result" + str(self.max_validation_result))
        print("test result" + str(self.test_result))
        exit("It's time to stop")

    def __error_analysis__(self):
        evaluate_results_valid = self._estimator.evaluate(
            input_fn=lambda: self._dataset.get_next(self._test_paths, training=False))
        target_list, predicts = self.process_evaluate_results(evaluate_results_valid)
        accuracy, precisions, recalls, fscores = _get_scores(target_list, predicts, self._leaf_indexes)
        print('this is the scores ', accuracy, precisions, recalls, fscores)
        from utilities.score_by_rank import evaluate_on_former_layer
        index2cause = self._dataset._cause_helper.get_index2cause()
        cause_tree = self._dataset._cause_helper._cause_tree
        evaluate_on_former_layer(lambda x: index2cause[x], cause_tree, predicts, target_list, self._leaf_indexes,
                                 self._cause2num)


class EvalCheckPointSaverListener(EvalCheckPointSaverListenerBase):
    _EOS = -1

    def __init__(self, params, estimator, dataset, valid_paths, test_paths=None, analysis=False, cause2num=None):
        '''
        由于目前采取的是钟皓曦的策略，所以我就不用early stop 什么的了
        '''
        if isinstance(dataset, DataSet_v2):
            self._dataset = dataset
        else:
            raise ('wrong type')

        super(EvalCheckPointSaverListener, self).__init__(params, estimator, dataset, valid_paths, test_paths = test_paths, analysis = analysis, cause2num = cause2num)
        self._EOS = params['EOS']

    def process_evaluate_results(self, evaluate_results, print_comparison=True):
        target_list = evaluate_results['predicted_and_cause_ids'][:, -1, :].tolist()
        predicts = evaluate_results['predicted_and_cause_ids'][:, 0, :].tolist()
        if print_comparison:
            _print_comparison(target_list, predicts)
        target_list = _cut_down(target_list, self._EOS, 1)
        predicts = _cut_down(predicts, self._EOS, 1)
        return target_list, predicts


class EvalCheckPointSaverListener2Baselines(EvalCheckPointSaverListenerBase):
    def process_evaluate_results(self, evaluate_results):
        target_list = evaluate_results['predicted_and_cause_ids'][:, 1].tolist()
        predicts = evaluate_results['predicted_and_cause_ids'][:, 0].tolist()
        return target_list, predicts


def _print_comparison(target, prediction):
    target = np.expand_dims(target, -2)
    prediction = np.expand_dims(prediction, -2)
    contrast = np.concatenate([target, prediction], axis=1)
    print(contrast[:20])


def _cut_down(labels, EOS, mode=0):
    '''
    keep labels before the first EOS was met
    labels shape=[batch_size, num_steps]
    mode 0: return original labels without redundant part after EOS shape = [batch_size, remain_cause_num]
    mode 1: return only the leaf labels shape = [batch_size]
    '''
    labels_prime = []
    if mode == 0:
        for steps in labels:
            steps_prime = []
            for step in steps:
                steps_prime.append(step)
                if step == EOS:
                    break
            labels_prime.append(steps_prime)
    elif mode == 1:
        for steps in labels:
            for i, step in enumerate(steps):
                if step == EOS or i == len(steps) - 1:
                    labels_prime.append(steps[i - 1])
                    break
    else:
        raise ('not supported mode')
    return labels_prime


def _get_scores(target_list, predicts, leaves):
    accuracy = accuracy_score(target_list, predicts)
    precisions, recalls, fscores, _ = precision_recall_fscore_support(target_list, predicts, labels=list(leaves),
                                                                      average='macro')
    return (accuracy, precisions, recalls, fscores)


def print_confusion_matrix(confusion, lables, trans_func, rate=0.6, mode=0):
    '''mode == 0: confusion matrix; mode == 1, print all charge's accuracy'''
    sum_array = np.sum(confusion, 1)
    # confusion /= sum_array[:np.newaxis]
    # print("%7s\t" % "", end="")
    # for lable in lables:
    #     print("%7s\t" % lable, end="")
    print("")
    if mode == 0:
        for i, array in enumerate(confusion):
            if sum_array[i] == 0:
                continue
            if array[i] / sum_array[i] > rate:
                continue
            print("label %s confused\t\t" % trans_func(lables[i]), end="")
            for j, ele in enumerate(array):
                if ele / sum_array[i] > 0.1:
                    print("%s  %.5f\t" % (trans_func(lables[j]), ele / sum_array[i]), end="")
            print('')
    elif mode == 1:
        for i, array in enumerate(confusion):
            if sum_array[i] == 0:
                print("charge %s doesn't have any test sample" % trans_func(lables[i]))
                continue
            print("label %s confused\t\t" % trans_func(lables[i]), end="")
            for j, ele in enumerate(array):
                if ele / sum_array[i] > 0.1:
                    print("%s  %.5f\t" % (trans_func(lables[j]), ele / sum_array[i]), end="")
            print('')
    else:
        raise NotImplementedError('errror')
    print("")
