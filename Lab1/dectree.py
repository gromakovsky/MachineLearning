import itertools
from functools import partial
from typing import Callable

from dataset import Label, DataSet, Item, change_dataset
from quality import QualityF


class DecisionTree:
    def __init__(self, value, avg_features=None, left=None, right=None, split_feature_idx=None):
        self._value = value
        self._avg_features = avg_features
        self._left = left
        self._right = right
        self._split_feature_idx = split_feature_idx

    @property
    def is_leaf(self):
        return self._value is not None

    @property
    def value(self):
        return self._value

    @property
    def left(self):
        return self._left

    @property
    def right(self):
        return self._right

    def print(self, indentation=0):
        if self.is_leaf:
            print('_' * indentation + str(self._value))
        else:
            to_print = 'f_{} > {}'.format(self._split_feature_idx, self._avg_features[self._split_feature_idx])
            self._left.print(indentation + len(to_print))
            print('_' * indentation + to_print)
            self._right.print(indentation + len(to_print))

    def classify(self, features):
        if self.is_leaf:
            return self._value
        else:
            return self._right.classify(features) if self._check(features) else self._left.classify(features)

    def prune(self):
        self._avg_features = None
        self._left = None
        self._right = None
        self._split_feature_idx = None

    def _check(self, features) -> bool:
        return features[self._split_feature_idx] > self._avg_features[self._split_feature_idx]


# Note: assuming that all items have the same number of features
class DecisionTreeBuilder:

    def __init__(self, train_data: DataSet, quality_function: QualityF):
        self._train_data = train_data
        self._quality_function = quality_function
        self._features_count = len(train_data.items[0].features)
        self._sum_features = [0] * self._features_count
        for item in train_data.items:
            assert isinstance(item, Item)
            for i in range(self._features_count):
                self._sum_features[i] += item.features[i]

        self._avg_features = [x / float(len(train_data)) for x in self._sum_features]

    def _build_decision_tree_(self, train_data: DataSet):
        all_the_same = train_data.all_the_same()
        if all_the_same is not None:
            return DecisionTree(all_the_same)

        max_split_quality = 0
        best_idx = 0
        for split_feature_idx in range(self._features_count):
            split_f = lambda d: d.features[split_feature_idx] > self._avg_features[split_feature_idx]
            split_quality = self._quality_function(train_data, split_f)
            assert split_quality >= 0
            if split_quality > max_split_quality:
                max_split_quality = split_quality
                best_idx = split_feature_idx

        split_f = lambda d: d.features[best_idx] > self._avg_features[best_idx]
        left_ds = change_dataset(train_data, partial(itertools.filterfalse, split_f))
        right_ds = change_dataset(train_data, partial(filter, split_f))
        left_tree = self._build_decision_tree_(left_ds)
        right_tree = self._build_decision_tree_(right_ds)
        return DecisionTree(None, self._avg_features, left_tree, right_tree, best_idx)

    def build(self):
        return self._build_decision_tree_(self._train_data)


def count_if(tree: DecisionTree, predicate: Callable[[Label], bool]):
    if tree.value is not None:
        return predicate(tree.value)
    else:
        return count_if(tree.left, predicate) + count_if(tree.right, predicate)


def majority(tree) -> int:
    return 1 if count_if(tree, lambda x: x == 1) > count_if(tree, lambda x: x == -1) else -1


class DecisionTreePruner:
    def __init__(self, tree: DecisionTree, valid_data):
        self.tree = tree
        self.valid_data = valid_data

    def _prune_(self, child):
        mistakes = test_decision_tree(self.tree, self.valid_data)
        child.value = majority(child)
        new_mistakes = test_decision_tree(self.tree, self.valid_data)
        print(mistakes, new_mistakes)
        if mistakes >= new_mistakes:
            child.prune()
        else:
            child.value = None
            self._prune_(child.left)
            self._prune_(child.right)

    def prune(self):
        self._prune_(self.tree)


def prune_decision_tree(tree, valid_data):
    DecisionTreePruner(tree, valid_data).prune()


def test_decision_tree(tree: DecisionTree, test_data: DataSet):
    mistakes = 0
    for item in test_data:
        assert isinstance(item, Item)
        expected = item.label
        calculated = tree.classify(item.features)
        mistakes += (expected != calculated)

    return mistakes


def test_decision_tree_and_print(tree, test_data):
    mistakes = test_decision_tree(tree, test_data)
    print('Tested decision tree on data of size {}, it made {} mistakes ({}%)'.format(len(test_data),
                                                                                      mistakes,
                                                                                      100. * mistakes / len(test_data))
)
