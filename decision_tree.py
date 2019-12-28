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


x = add_indices(x) #Data should must have a column which stores serial number of the instances.


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

    def partitioning(self,data):
        partitions = []
        column_elements = []
        for i in range(data.shape[1] - 1):
            for j in data[:, i + 1]:
                if j in column_elements:
                    continue
                else:
                    column_elements.append(j)
            partitions.append(column_elements)
            column_elements = []

        return partitions

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
            entropy += ((class_count[key] / len(data)) * -np.log2(round(class_count[key] / len(data), 5)))

        return entropy

    def total_entropy(self, data_below, data_above):
        total_instances = len(data_above) + len(data_below)
        total_entropy = (len(data_below)/total_instances * self.entropy(data_below)) + (len(data_above)/total_instances * self.entropy(data_above))

        return  total_entropy

    def best_partitioning(self, data, partitioning):
        minimun_entropy = 20
        for i in range(len(partitioning)):
            for value in partitioning[i]:
                data_below, data_above = self.split(data, i, value)
                total_entropy = self.total_entropy(data_below, data_above)
                if(total_entropy < minimun_entropy):
                    minimun_entropy = total_entropy
                    column = i
                    partition_value = value

        return column, partition_value

    def grow(self):
        pass


obj = DecisionTree(np.array([[0,2,3,4,5],[1,2,6,7.5,4],[2,3,6,9.7,3]]), np.array([0,0,1]))
c1, c2 = obj.split(x, 2, 3)
en = obj.entropy(np.array([[0,2,3,4,5],[1,2,6,7.5,4],[2,3,6,9.7,3]]))
g = obj.partitioning(np.array([[0,2,3,4,5],[1,2,6,7.5,4],[2,3,6,9.7,3]]))
c,v = obj.best_partitioning(np.array([[0,2,3,4,5],[1,2,6,7.5,4],[2,3,6,9.7,3]]),g)
