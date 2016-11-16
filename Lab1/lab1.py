from ml.classifier import Classifier, test_classifier
from ml.dataset import DataSet, read_data
from ml.forest import RandomForest
from ml.quality import information_gain, gini_gain
from ml.selection import order_features, estimate_feature_selection, RFImportanceCalculator, PearsonImportanceCalculator

TRAIN_DATA_NAME = 'data/arcene_train.data'
TRAIN_LABELS_NAME = 'data/arcene_train.labels'
VALID_DATA_NAME = 'data/arcene_valid.data'
VALID_LABELS_NAME = 'data/arcene_valid.labels'


def test_classifier_and_print(classifier: Classifier, test_data: DataSet):
    mistakes = test_classifier(classifier, test_data)
    print('Tested classifier on data of size {}, it made {} mistakes ({}%)'.format(len(test_data),
                                                                                   mistakes,
                                                                                   100. * mistakes / len(test_data)))


def run_tests(classifier, train_data, valid_data):
    print('Testing classifier on train data…')
    test_classifier_and_print(classifier, train_data)
    print('Testing classifier on validation data…')
    test_classifier_and_print(classifier, valid_data)


def main():
    train_data = read_data(TRAIN_DATA_NAME, TRAIN_LABELS_NAME)
    validation_data = read_data(VALID_DATA_NAME, VALID_LABELS_NAME)

    quality_functions = {
        'Information gain': information_gain,
        'Gini gain': gini_gain,
    }

    for quality_function_name, quality_function in quality_functions.items():
        print('Using', quality_function_name)
        print('Building random forest…')
        forest = RandomForest(train_data, quality_function, trees_num=50)
        importance_calculators = {
            'random forest': RFImportanceCalculator(forest),
            'Pearson': PearsonImportanceCalculator(),
        }
        for importance_calculator_name, importance_calculator in importance_calculators.items():
            print('Using', importance_calculator_name)
            print('Ordering features…')
            ordered_features = order_features(importance_calculator, train_data)
            print('Estimating feature selection…')
            estimate_feature_selection(ordered_features, train_data, validation_data)

if __name__ == '__main__':
    main()
