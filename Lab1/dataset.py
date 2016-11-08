class Item:

    def __init__(self, features: [int], label: int):
        self.features = features
        self.label = label


class DataSet:

    def __init__(self, items: [Item]):
        self.items = items


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
