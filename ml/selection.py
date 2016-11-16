from operator import itemgetter
from typing import List

from ml.dataset import DataSet, shuffle_feature
from ml.forest import RandomForest
from ml.svm import test_svm_classifier


def calc_feature_importance(forest: RandomForest, train_data: DataSet) -> List[float]:
    main_error = forest.oob_error(train_data)
    print('OOB error: {}'.format(main_error))
    res = []
    for i in range(train_data.features_num):
        shuffled = shuffle_feature(train_data, i)
        shuffled_error = forest.oob_error(shuffled)
        res.append(sum(me - se for me, se in zip(main_error, shuffled_error)))

    return res


def order_features(forest: RandomForest, train_data: DataSet) -> List[int]:
    importance = calc_feature_importance(forest, train_data)
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
