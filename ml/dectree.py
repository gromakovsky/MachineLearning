import itertools
from functools import partial
from typing import Callable, List, Optional

from ml.classifier import Classifier, test_classifier
from ml.dataset import Feature, Label, DataSet, Item, change_dataset
from ml.quality import QualityF


class DecisionTree(Classifier):

    def __init__(self, value: Label, avg_features=None, left=None, right=None, split_feature_idx=None):
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

    def classify(self, features: List[Feature]) -> Label:
        if self.is_leaf:
            return self._value
        else:
            return self._right.classify(features) if self._check(features) else self._left.classify(features)

    def prune(self, valid_data: DataSet):
        self._prune(valid_data=valid_data, root=self)

    def _prune(self, valid_data: DataSet, root):
        if self.is_leaf:
            return

        mistakes = test_classifier(root, valid_data)
        self._value = self._majority()
        new_mistakes = test_classifier(root, valid_data)
        print(mistakes, new_mistakes)
        if mistakes >= new_mistakes:
            self._become_leaf()
        else:
            self._value = None
            self._left._prune(valid_data, root)
            self._right._prune(valid_data, root)

    def _become_leaf(self):
        self._avg_features = None
        self._left = None
        self._right = None
        self._split_feature_idx = None

    def _check(self, features) -> bool:
        return features[self._split_feature_idx] > self._avg_features[self._split_feature_idx]

    def _count_if(self, predicate: Callable[[Label], bool]) -> int:
        if self.is_leaf:
            return int(predicate(self._value))
        else:
            return self._left._count_if(predicate) + self._right._count_if(predicate)

    def _majority(self) -> Label:
        return 1 if self._count_if(lambda x: x == 1) > self._count_if(lambda x: x == -1) else -1


def build_decision_tree(train_data: DataSet, quality_function: QualityF,
                        features_to_use: Optional[List[int]]=None) -> DecisionTree:
    return _DecisionTreeBuilder(train_data, quality_function, features_to_use).build()


##########################################################
# Implementation
##########################################################

# Note: assuming that all items have the same number of features
class _DecisionTreeBuilder:

    def __init__(self, train_data: DataSet, quality_function: QualityF, features_to_use: Optional[List[int]]):
        self._train_data = train_data
        self._quality_function = quality_function
        self._features_count = train_data.features_num
        self._features_to_use = features_to_use
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
        features_to_use = self._features_to_use if self._features_to_use is not None else range(self._features_count)
        for split_feature_idx in features_to_use:
            split_f = lambda d: d.features[split_feature_idx] > self._avg_features[split_feature_idx]
            split_quality = self._quality_function(train_data, split_f)
            if split_quality > max_split_quality:
                max_split_quality = split_quality
                best_idx = split_feature_idx

        split_f = lambda d: d.features[best_idx] > self._avg_features[best_idx]
        left_ds = change_dataset(train_data, partial(itertools.filterfalse, split_f))
        right_ds = change_dataset(train_data, partial(filter, split_f))
        left_tree = self._build_decision_tree_(left_ds)
        right_tree = self._build_decision_tree_(right_ds)
        return DecisionTree(None, self._avg_features, left_tree, right_tree, best_idx)

    def build(self) -> DecisionTree:
        return self._build_decision_tree_(self._train_data)
