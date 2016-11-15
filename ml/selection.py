from typing import List

from ml.dataset import DataSet, shuffle_feature
from ml.forest import RandomForest


def select_features(forest: RandomForest, train_data: DataSet) -> List[float]:
    main_error = forest.oob_error(train_data)
    print('OOB error: {}'.format(main_error))
    res = []
    for i in range(train_data.features_num):
        shuffled = shuffle_feature(train_data, i)
        shuffled_error = forest.oob_error(shuffled)
        res.append(sum(me - se for me, se in zip(main_error, shuffled_error)))

    return res
