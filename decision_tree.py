import numpy as np
from sklearn import datasets

dataset = datasets.load_iris()
x = dataset.data
y = dataset.target


def add_indices(data):
    new_data = []
    rows = data.shape[0]
    for i in range(rows):
        sample = np.append(i, data[i])
        new_data.append(list(sample))

    print(new_data)
    return np.array(new_data)


x = add_indices(x)


def train_test_split(input, target, ratio):
    shuffle_indices = np.random.permutation(input.shape[0])
    test_size = int(input.shape[0] * ratio)
    train_indices = shuffle_indices[:test_size]
    test_indices = shuffle_indices[test_size:]
    xTrain = input[train_indices]
    yTrain = target[train_indices]
    xTest = input[test_indices]
    yTest = target[test_indices]
    return xTrain, yTrain, xTest, yTest


xTrain, yTrain, xTest, yTest = train_test_split(x, y, 0.8)


class DecisionTree:
    def __init__(self, training_data, target, impurity_measure='entropy'):
        self.data = training_data
        self.impurity_measure = impurity_measure
        self.target = target
        print(self.purity(training_data))

    def purity(self, data):
        diff_classes = {}
        self.data = data
        indices = data[:, 0]
        class_in_sample = [self.target[int(i)] for i in indices]
        for i in range(len(class_in_sample)):
            if i == 0:
                diff_classes[class_in_sample[i]] = 1
            if class_in_sample[i] in diff_classes.keys():
                diff_classes[class_in_sample[i]] += 1
            else:
                return 0

        return 1

    def split(self, data, column, value):
        self.data = data
        data_above = []
        data_below = []
        for i in range(len(data)):
            if data[i, column] > value:
                data_above.append(list(data[i]))
            else:
                data_below.append(list(data[i]))

        return np.array(data_below), np.array(data_above)

    def entropy(self, data):
        class_count = {}
        entropy = 0
        resultant_target = [self.target[int(i)] for i in data[:, 0]]
        for i in range(len(resultant_target)):
            if resultant_target[i] in class_count.keys():
                class_count[resultant_target[i]] += 1
            else:
                class_count[resultant_target[i]] = 1

        for key in class_count:
            print(class_count[key])
            entropy += class_count[key] / len(data)

        return entropy

    def total_entropy(self):
        pass

    def grow(self):
        pass


obj = DecisionTree(x, y)
c1, c2 = obj.split(x, 2, 3)
en = obj.entropy(x)
