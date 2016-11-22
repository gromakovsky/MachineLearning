from collections import namedtuple
from operator import itemgetter
from typing import List, Tuple
import numpy as np


fst = itemgetter(0)


# data is array of objects, were each object is array of features
def pca_eigens(arr: np.ndarray) -> List[Tuple[float, np.ndarray]]:
    items_num, dimensionality = arr.shape
    transposed = arr.transpose()
    assert transposed.shape == (dimensionality, items_num)
    mean_values = np.array([[np.mean(i)] for i in transposed])
    assert mean_values.shape == (dimensionality, 1)

    scatter_matrix = np.zeros((dimensionality, dimensionality))
    for i in range(items_num):
        d = arr[i].reshape(dimensionality, 1) - mean_values
        scatter_matrix += d @ d.T

    assert scatter_matrix.shape == (dimensionality, dimensionality)

    eigen_values, eigen_vectors = np.linalg.eig(scatter_matrix)
    eigen_pairs = [(abs(eigen_values[i]), eigen_vectors[:, i]) for i in range(dimensionality)]
    eigen_pairs.sort(key=fst, reverse=True)
    return eigen_pairs


BrokenStickRes = namedtuple('BrokenStickRes', ('k', 'normalized', 'steps'))


def broken_stick(eigens: List[Tuple[float, np.ndarray]]) -> BrokenStickRes:
    s = sum(map(fst, eigens))
    n = len(eigens)
    normalized = [i / s for i in map(fst, eigens)]
    steps = []
    res = None
    for i in range(len(normalized)):
        l = sum(1 / (j + 1) for j in range(i, n)) / n
        steps.append(l)
        if normalized[i] < l and res is None:
            res = i

    return BrokenStickRes(k=res, normalized=normalized, steps=steps)


def transformation_matrix(eigens: List[np.ndarray], k: int) -> np.ndarray:
    assert isinstance(eigens, list)
    assert len(eigens) > k
    assert k > 0
    res = np.array(list(map(list, eigens[:k])))
    assert res.shape == (k, len(eigens))
    return res
