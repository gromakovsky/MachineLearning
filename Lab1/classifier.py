import abc
from typing import List

from dataset import DataSet, Feature, Item, Label


class Classifier:

    @abc.abstractmethod
    def classify(self, features: List[Feature]) -> Label:
        raise NotImplemented


def test_classifier(tree: Classifier, test_data: DataSet):
    mistakes = 0
    for item in test_data.items:
        assert isinstance(item, Item)
        expected = item.label
        calculated = tree.classify(item.features)
        mistakes += expected != calculated

    return mistakes
