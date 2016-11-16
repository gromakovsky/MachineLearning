from ml.classifier import Classifier, test_classifier
from ml.dataset import DataSet, read_data
# import ml.dectree
from ml.forest import RandomForest
from ml.quality import information_gain, gini_gain
from ml.selection import calc_feature_importance

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
        print('Selecting features…')
        features_importance = calc_feature_importance(forest, train_data)
        print('Importance: {}'.format(features_importance))
        # run_tests(forest, train_data, valid_data)
         # print('Building decision tree…')
        # tree = dectree.build_decision_tree(train_data, quality_function)
        # run_tests(tree, train_data, valid_data)
        # print('Pruning decision tree…')
        # tree.prune(valid_data)
        # tree.print()

if __name__ == '__main__':
    main()
