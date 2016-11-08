from typing import List
from math import sqrt
import random

from dataset import DataSet, Item
from quality import QualityF

from dectree import build_decision_tree


class DecisionTreeMeta:

    def __init__(self, training_data: DataSet, used_features: List[int]):
        self._training_data = training_data
        self._used_features = used_features


class RandomForest:

    def __init__(self, train_data: DataSet, quality_function: QualityF, trees_num: int):
        self._trees = []
        for _ in range(trees_num):
            f_num = train_data.features_num
            features_to_use = list(self._random_indices(f_num, self._features_to_use(f_num)))
            items = list(self._random_items(train_data.items, features_to_use))
            sub_ds = DataSet(items)
            tree = build_decision_tree(sub_ds, quality_function)
            meta = DecisionTreeMeta(training_data=sub_ds, used_features=features_to_use)
            self._trees.append((tree, meta))

    @staticmethod
    def _features_to_use(features_num):
        return int(sqrt(features_num))

    @staticmethod
    def _random_items(items: List[Item], features_to_use):
        n = len(items)
        for _ in range(n):
            i = items[random.randrange(n)]
            yield Item(features=i.all_features, label=i.label, features_to_use=features_to_use)

    @staticmethod
    def _random_indices(max_val, n):
        for _ in range(n):
            yield random.randrange(max_val)
