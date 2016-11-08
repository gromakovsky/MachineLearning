#!/usr/bin/env python

from dataset import read_data

TRAIN_DATA_NAME = 'data/arcene_train.data'
TRAIN_LABELS_NAME = 'data/arcene_train.labels'
VALID_DATA_NAME = 'data/arcene_valid.data'
VALID_LABELS_NAME = 'data/arcene_valid.labels'


def main():
    train_data = read_data(TRAIN_DATA_NAME, TRAIN_LABELS_NAME)
    print(train_data.items)


if __name__ == '__main__':
    main()
