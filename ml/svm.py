from typing import List, Optional
from sklearn import svm

from ml.classifier import Classifier, test_classifier
from ml.dataset import DataSet, Feature, Label


class SvmClassifier(Classifier):

    def __init__(self, data: DataSet, features_to_use: Optional[List[int]]):
        self._data = data
        self._features_to_use = features_to_use if features_to_use is not None else list(range(data.features_num))
        self._classifier = svm.SVC(kernel='linear')
        self._classifier.fit(*zip(*(self._project_features(item.features), item.label for item in data.items)))

    def classify(self, features: List[Feature]) -> Label:
        return self._classifier.predict(self._project_features(features))

    def _project_features(self, features: List[Feature]) -> List[Feature]:
        return [features[i] for i in self._features_to_use]


def test_svm_classifier(train_data: DataSet, features_to_use: List[int], validation_data: DataSet) -> int:
    classifier = SvmClassifier(train_data, features_to_use)
    return test_classifier(classifier, validation_data)
