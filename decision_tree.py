import numpy as np
from sklearn import datasets

dataset = datasets.load_iris()
x = dataset.data
y = dataset.target

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
        self.data = self.add_indices(training_data)
        self.impurity_measure = impurity_measure
        self.target = target
        self.data = np.append(self.data, target.reshape(len(self.data), 1), axis = 1)
        print(self.data)
        print(self.purity(self.data))

    def add_indices(self,data):
        new_data = []
        rows = data.shape[0]
        for i in range(rows):
            sample = np.append(i, data[i])
            new_data.append(list(sample))

        return np.array(new_data)

    def purity(self, data):
        self.data = data
        class_in_sample = data[:,-1]
        first_class = class_in_sample[0]
        for i in class_in_sample:
            if(i != first_class):
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
        target = data[:,-1]
        for i in range(len(target)):
            if target[i] in class_count.keys():
                class_count[target[i]] += 1
            else:
                class_count[target[i]] = 1

        for key in class_count:
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
                data_below, data_above = self.split(data, i+1, int(value))
                if data_below.size != 0 and data_above.size != 0:
                    print("data below is -- {}, data above is -- {}".format(data_below, data_above))
                    print("entropy is -- {}".format(self.total_entropy(data_below, data_above)))
                    total_entropy = self.total_entropy(data_below, data_above)

                    if(total_entropy < minimun_entropy):
                        minimun_entropy = total_entropy
                        column = i
                        partition_value = value

            print(minimun_entropy, column, partition_value)

    def grow(self):
        pass

xT = np.array([[2,3,4,5],[2,6,7.5,4],[3,6,9.7,3]])
yT = np.array([0,0,1])
obj = DecisionTree(xT, yT)
# par = obj.partitioning(np.append(xT, yT.reshape(len(xT), 1), axis = 1))
# db,da = obj.split(np.append(xT, yT.reshape(len(xT), 1), axis = 1),1,2)
# en = obj.entropy(np.append(xT, yT.reshape(len(xT), 1), axis = 1))
# en2 = obj.total_entropy(db,da)
# obj.best_partitioning(np.append(xT, yT.reshape(len(xT), 1), axis = 1), par)




