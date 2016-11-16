import abc
from operator import itemgetter
from typing import List
from scipy import stats
from sklearn.metrics import mutual_info_score

from ml.dataset import DataSet, shuffle_feature
from ml.forest import RandomForest
from ml.svm import test_svm_classifier


class ImportanceCalculator:

    @abc.abstractmethod
    def __call__(self, train_data: DataSet) -> List[float]:
        raise NotImplemented


class RFImportanceCalculator(ImportanceCalculator):

    def __init__(self, forest: RandomForest):
        self._forest = forest

    def __call__(self, train_data: DataSet) -> List[float]:
        main_error = self._forest.oob_error(train_data)
        print('OOB error: {}'.format(main_error))
        res = []
        for i in range(train_data.features_num):
            shuffled = shuffle_feature(train_data, i)
            shuffled_error = self._forest.oob_error(shuffled)
            res.append(sum(me - se for me, se in zip(main_error, shuffled_error)))

        return res


class PearsonImportanceCalculator(ImportanceCalculator):

    def __call__(self, train_data: DataSet) -> List[float]:
        labels = [item.label for item in train_data.items]
        features_num = train_data.features_num
        features_transposed = [[item.features[i] for item in train_data.items] for i in range(features_num)]
        return [stats.pearsonr(features_transposed[i], labels)[0] for i in range(features_num)]


class SpearmanImportanceCalculator(ImportanceCalculator):

    def __call__(self, train_data: DataSet) -> List[float]:
        labels = [item.label for item in train_data.items]
        features_num = train_data.features_num
        features_transposed = [[item.features[i] for item in train_data.items] for i in range(features_num)]
        return [stats.spearmanr(features_transposed[i], labels)[0] for i in range(features_num)]


class MutualInformationImportanceCalculator(ImportanceCalculator):

    def __call__(self, train_data: DataSet) -> List[float]:
        labels = [item.label for item in train_data.items]
        features_num = train_data.features_num
        features_transposed = [[item.features[i] for item in train_data.items] for i in range(features_num)]
        return [mutual_info_score(features_transposed[i], labels) for i in range(features_num)]


def order_features(calculator: RFImportanceCalculator, train_data: DataSet) -> List[int]:
    importance = calculator(train_data)
    # print('Importance: {}'.format(importance))
    indices = list(range(train_data.features_num))
    return list(map(itemgetter(0), sorted(zip(indices, importance), key=itemgetter(1), reverse=True)))


def estimate_feature_selection(ordered_features: List[int],
                               train_data: DataSet, validation_data: DataSet) -> List[float]:
    validation_items_num = len(validation_data.items)
    print('Testing SVM…')
    step = 30
    finish = 1500
    for n in range(step, finish, step):
        features_to_use = ordered_features[:n]
        errors_num = test_svm_classifier(train_data, features_to_use, validation_data)
        accuracy = (validation_items_num - errors_num) / validation_items_num
        print('Number of features: {}, accuracy: {}'.format(n, accuracy))
