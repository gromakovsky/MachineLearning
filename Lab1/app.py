#!/usr/bin/env python

from dataset import read_data
from quality import information_gain
import dectree

TRAIN_DATA_NAME = 'data/arcene_train.data'
TRAIN_LABELS_NAME = 'data/arcene_train.labels'
VALID_DATA_NAME = 'data/arcene_valid.data'
VALID_LABELS_NAME = 'data/arcene_valid.labels'


def main():
    train_data = read_data(TRAIN_DATA_NAME, TRAIN_LABELS_NAME)

    quality_functions = {
        'IGain': information_gain,
    }
    for quality_function_name, quality_function in quality_functions.items():
        print('Using', quality_function_name)
        tree_builder = dectree.DecisionTreeBuilder(train_data, quality_function)
        print('Building decision tree...')
        tree = tree_builder.build()
        tree.print()

if __name__ == '__main__':
    main()
