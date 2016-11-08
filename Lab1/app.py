#!/usr/bin/env python

from dataset import read_data
from quality import information_gain, gini_gain
import dectree

TRAIN_DATA_NAME = 'data/arcene_train.data'
TRAIN_LABELS_NAME = 'data/arcene_train.labels'
VALID_DATA_NAME = 'data/arcene_valid.data'
VALID_LABELS_NAME = 'data/arcene_valid.labels'


def test_decision_tree_and_print(tree, test_data):
    mistakes = dectree.test_decision_tree(tree, test_data)
    print('Tested decision tree on data of size {}, it made {} mistakes ({}%)'.format(len(test_data),
                                                                                      mistakes,
                                                                                      100. * mistakes / len(test_data))
)


# def run_tests(tree, train_data, valid_data):
#     print('Testing decision tree on train data…')
#     test_decision_tree_and_print(tree, train_data)
#     print('Testing decision tree on valid data…')
#     test_decision_tree_and_print(tree, valid_data)


def main():
    train_data = read_data(TRAIN_DATA_NAME, TRAIN_LABELS_NAME)
    valid_data = read_data(VALID_DATA_NAME, VALID_LABELS_NAME)

    quality_functions = {
        'Information gain': information_gain,
        'Gini gain': gini_gain,
    }
    for quality_function_name, quality_function in quality_functions.items():
        print('Using', quality_function_name)
        print('Building decision tree…')
        tree = dectree.build_decision_tree(train_data, quality_function)
        # tree.print()
        # run_tests(tree, train_data, valid_data)
        print('Pruning decision tree…')
        tree.prune(valid_data)
        tree.print()
        # run_tests(tree, train_data, valid_data)

if __name__ == '__main__':
    main()
