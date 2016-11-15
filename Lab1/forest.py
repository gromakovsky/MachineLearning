from functools import partial
from math import sqrt
from operator import attrgetter
import random
from typing import List, Iterable

from classifier import Classifier
from dataset import Feature, DataSet, Item, Label
from dectree import build_decision_tree, DecisionTree
from quality import QualityF


class DecisionTreeMeta:

    def __init__(self, training_data: DataSet, item_indices: Iterable[int], used_features: List[int]):
        self._training_data = training_data
        self._item_indices = set(item_indices)
        self._used_features = used_features

    @property
    def item_indices(self):
        return self._item_indices


class ExtendedDecisionTree:

    def __init__(self, tree: DecisionTree, meta: DecisionTreeMeta):
        self._tree = tree
        self._meta = meta

    @property
    def tree(self):
        return self._tree

    @property
    def meta(self):
        return self._meta


class RandomForest(Classifier):

    def __init__(self, train_data: DataSet, quality_function: QualityF, trees_num: int):
        self._trees = []
        for _ in range(trees_num):
            f_num = train_data.features_num
            features_to_use = list(self._random_indices(f_num, self._features_to_use_num(f_num)))
            item_indices, items = zip(*self._random_items(train_data.items))
            sub_ds = DataSet(items)
            tree = build_decision_tree(sub_ds, quality_function, features_to_use)
            meta = DecisionTreeMeta(training_data=sub_ds, item_indices=item_indices, used_features=features_to_use)
            self._trees.append(ExtendedDecisionTree(tree, meta))

    def classify(self, features: List[Feature]) -> Label:
        return self._classify(map(attrgetter('tree'), self._trees), features)

    def oob_error(self, train_data: DataSet) -> List[float]:
        res = []
        for extended_tree in self._trees:
            total_classifications = 0
            total_errors = 0
            for i, item in enumerate(train_data.items):
                if i in extended_tree.meta.item_indices:
                    continue

                total_classifications += 1
                total_errors += int(item.label != extended_tree.tree.classify(item.features))

            print(total_classifications, total_errors)
            res.append(float(total_errors) / total_classifications)

        return res

    @staticmethod
    def _classify(trees: Iterable[DecisionTree], features: List[Feature]):
        s = sum(map(partial(DecisionTree.classify, features=features), trees))
        return -1 if s < 0 else 1

    @staticmethod
    def _features_to_use_num(features_num):
        return int(sqrt(features_num))

    @staticmethod
    def _random_items(items: List[Item]):
        n = len(items)
        for _ in range(n):
            i = random.randrange(n)
            yield i, items[i]

    @staticmethod
    def _random_indices(max_val, n):
        for _ in range(n):
            yield random.randrange(max_val)
