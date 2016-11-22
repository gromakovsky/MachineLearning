from typing import List
import numpy as np


def pca(data: List[List[float]]):
    arr = np.array(data)
    assert arr.shape == (1000, 200)
