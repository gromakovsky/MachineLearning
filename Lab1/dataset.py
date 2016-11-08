from typing import Callable, Iterable, List, Optional


Label = int
Feature = int


class Item:

    def __init__(self, features: List[Feature], label: Label):
        self._features = features
        self._label = label

    @property
    def features(self) -> List[Feature]:
        return self._features

    @property
    def label(self) -> Label:
        return self._label


class DataSet:

    def __init__(self, items: List[Item]):
        self.items = items

    @property
    def features_num(self):
        return len(self.items[0].features) if self.items else 0

    # returns 1 if all items have label=1, -1 if all items have label=-1, None otherwise
    def all_the_same(self) -> Optional[Label]:
        all_positive = True
        all_negative = True
        for item in self.items:
            assert isinstance(item, Item)
            if item.label == -1:
                all_positive = False
            else:
                all_negative = False

        if all_positive:
            return 1

        if all_negative:
            return -1

    def __len__(self):
        return len(self.items)


def change_dataset(data: DataSet, f: Callable[[List[Item]], Iterable[Item]]):
    return DataSet(list(f(data.items)))


def read_data(data_file_name, labels_file_name):
    items = []
    with open(data_file_name) as data_file:
        with open(labels_file_name) as labels_file:
            data_lines = data_file.readlines()
            labels_lines = labels_file.readlines()
            for i in range(len(data_lines)):
                features = [int(w) for w in data_lines[i].split(' ')[:-1]]
                label = int(labels_lines[i][:-1])
                items.append(Item(features, label))

    return DataSet(items)
