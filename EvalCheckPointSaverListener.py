import os, glob
import shutil
from sys import stdout
import tensorflow as tf
from sklearn.metrics import precision_recall_fscore_support, accuracy_score, confusion_matrix
import time
import numpy as np
from collections import namedtuple


class Result(namedtuple('Result', ('step', 'accuracy', 'precision', 'recall', 'fscore'))):
    '''we only collectes the macro value of these values'''
    pass


class EvalCheckpointSaverListener(tf.train.CheckpointSaverListener):
    results = []
    eval_step = 0
    esitmator = None
    ds_val = None
    params = None
    model_dir = None

    '''do test while max validation result appears'''
    max_validation_result = Result(0, 0, 0, 0, 0)
    max_test_result = Result(0, 0, 0, 0, 0)

    def __init__(self, estimator, ds_eval, ds_test, params, civil=False, test=False):
        self.estimator = estimator
        self.test = test
        self.ds_eval = ds_eval
        self.ds_test = ds_test
        self.params = params
        self.model_dir = self.estimator.model_dir
        self.optimal_model_dir = os.path.join(self.model_dir, 'optimal_model')
        # self.fit_binarizer()
        self.leaf_indexes = self.ds_eval.get_leaf_index()
        if civil:
            self.early_stop_steps = 30
            self.min_training_num = 240
        else:
            self.early_stop_steps = 20
            self.min_training_num = 140
        if not os.path.exists(self.optimal_model_dir):
            os.mkdir(self.optimal_model_dir)

    # def fit_binarizer(self):
    #     self.mlb = MultiLabelBinarizer()
    #     self.mlb.fit([[i for i in range(self.ds_eval.num_causes)]])

    def after_save(self, session, global_step):
        evaluate_results = self.estimator.evaluate(input_fn=lambda: self.ds_eval.get_data_set())
        self._evaluate(evaluate_results, self.params['EOS'])

    def __early_stop__(self):
        num_eval = len(self.results)

        if self.results[-1].fscore > self.max_validation_result.fscore:
            self.max_validation_result = self.results[-1]
            if num_eval < self.early_stop_steps:
                return
            evaluate_results_test = self.estimator.evaluate(input_fn=lambda: self.ds_test.get_data_set())
            target_list, predicts = self.process_evaluate_reusults(evaluate_results_test, self.params['EOS'],
                                                                   print_comparison=False)
            accuracy = accuracy_score(target_list, predicts, )
            precisions, recalls, fscores, _ = precision_recall_fscore_support(target_list, predicts,
                                                                              labels=self.leaf_indexes,
                                                                              average='macro')
            self.max_test_result = Result(self.eval_step, accuracy, precisions, recalls, fscores)
            latest_checkpoint = self.estimator.latest_checkpoint()
            shutil.rmtree(self.optimal_model_dir)
            os.mkdir(self.optimal_model_dir)
            for file in glob.glob(latest_checkpoint + '*'):
                shutil.copy(file, self.optimal_model_dir)
        # # 策略连续8个验证下降
        # for i in range(self.early_stop_steps - 1):
        #     if self.results[num_eval - self.early_stop_steps + i].fscore < self.results[
        #         num_eval - self.early_stop_steps + i + 1].fscore:
        #         return
        # 策略连续20个验证达不到最高
        if num_eval < self.min_training_num:
            return

        for i in range(self.early_stop_steps):
            if self.results[num_eval - i - 1].fscore >= self.max_validation_result.fscore:
                return
        print("max validation result" + str(self.max_validation_result))
        print("max test result" + str(self.max_test_result))
        exit("It's time to stop")

    def process_evaluate_reusults(self, evaluate_results, EOS, print_comparison=True):
        target_list = evaluate_results['predicted_and_cause_ids'][:, -1, :].tolist()
        predicts = evaluate_results['predicted_and_cause_ids'][:, 0, :].tolist()
        if print_comparison:
            self.print_comparison(target_list, predicts)
        target_list = self._cut_down(target_list, EOS, 1)
        predicts = self._cut_down(predicts, EOS, 1)
        return target_list, predicts

    def _evaluate(self, evaluate_results, EOS):
        self.eval_step += 1
        target_list, predicts = self.process_evaluate_reusults(evaluate_results, EOS)
        confusion = confusion_matrix(target_list, predicts, self.leaf_indexes)
        print(time.asctime(time.localtime(time.time())))
        accuracy = accuracy_score(target_list, predicts)
        print('accuracy score %f' % accuracy)
        precisions, recalls, fscores, _ = precision_recall_fscore_support(target_list, predicts,
                                                                          labels=self.leaf_indexes,
                                                                          average='macro')
        print('precsion  ' + str(precisions) + ' recall ' + str(recalls) + ' fscores ' + str(fscores))
        precisions, recalls, fscores, _ = precision_recall_fscore_support(target_list, predicts,
                                                                          average='macro')

        print('macro average precision recall f1-score: %f %f %f' %
              (precisions, recalls, fscores))
        print_confusion_matrix(confusion, self.leaf_indexes, self.ds_eval.cause_project_name, rate=fscores,
                               mode=self.test)
        self.results.append(Result(self.eval_step, accuracy, precisions, recalls, fscores))
        print('\n\n')
        self.__early_stop__()
        print('highest validation score' + str(self.max_validation_result))
        print('and the test score' + str(self.max_test_result))
        stdout.flush()

    def print_comparison(self, target, prediction):
        target = np.expand_dims(target, -2)
        prediction = np.expand_dims(prediction, -2)
        contrast = np.concatenate([target, prediction], axis=1)
        print(contrast[:20])

    def _cut_down(self, labels, EOS, mode=0):
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


def print_confusion_matrix(confusion, lables, trans_func, rate=0.6, mode=False):
    '''mode == 0: confusion matrix; mode == 1, print all charge's accuracy'''
    sum_array = np.sum(confusion, 1)
    # confusion /= sum_array[:np.newaxis]
    # print("%7s\t" % "", end="")
    # for lable in lables:
    #     print("%7s\t" % lable, end="")
    print("")
    if not mode:
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
    else:
        for i, array in enumerate(confusion):
            if sum_array[i] == 0:
                print("charge %s doesn't have any test sample" % trans_func(lables[i]))
                continue
            print("label %s confused\t\t" % trans_func(lables[i]), end="")
            for j, ele in enumerate(array):
                if ele / sum_array[i] > 0.1:
                    print("%s  %.5f\t" % (trans_func(lables[j]), ele / sum_array[i]), end="")
            print('')
    print("")