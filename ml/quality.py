import math
import itertools
from functools import partial
from typing import Callable

from ml.dataset import Item, DataSet, change_dataset


SplitF = Callable[[Item], bool]
QualityF = Callable[[DataSet, SplitF], float]


def entropy_binary(p: float) -> float:
    q = 1. - p
    return -p * math.log2(p) - q * math.log2(q) if p * q else 0


def data_entropy(data: DataSet) -> float:
    items = data.items
    total = len(items)
    positive_count = len(list(filter(lambda d: d.label == 1, items)))
    p_pos = positive_count / float(total)
    return entropy_binary(p_pos)


def information_gain(data: DataSet, split_func: SplitF) -> float:
    entropy = data_entropy(data)
    new_entropy = 0
    left_part = change_dataset(data, partial(itertools.filterfalse, split_func))
    right_part = change_dataset(data, partial(filter, split_func))
    for part in (left_part, right_part):
        assert isinstance(part, DataSet)
        part_len = len(part)
        new_entropy += float(part_len) / len(data) * data_entropy(part) if part_len else 0

    return entropy - new_entropy


def gini_gain(data: DataSet, split_func: SplitF) -> float:
    sum_squares = 0
    left_part = change_dataset(data, partial(itertools.filterfalse, split_func))
    right_part = change_dataset(data, partial(filter, split_func))
    for part in (left_part, right_part):
        assert isinstance(part, DataSet)
        part_len = len(part)
        sum_squares += (float(part_len) / len(data)) ** 2

    return 1 - sum_squares
