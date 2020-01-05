import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt

dataset = datasets.load_iris()
x = dataset.data
y = dataset.target

plt.scatter(x[:, 3], x[:, 2])
plt.xlim(0, 2.5)
plt.ylim(0, 7)
plt.savefig('/home/malay/PycharmProjects/DecisionTree/graph4.png')
plt.show()


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
        self.data = np.append(self.data, target.reshape(len(self.data), 1), axis=1)
        print(self.data)
        print(self.purity(self.data))
        self.counter = 0

    def train(self):
        return self.grow(self.data)

    @staticmethod
    def add_indices(data):
        new_data = []
        rows = data.shape[0]
        for i in range(rows):
            sample = np.append(i, data[i])
            new_data.append(list(sample))

        return np.array(new_data)

    def purity(self, data):
        self.data = data
        class_in_sample = data[:, -1]
        first_class = class_in_sample[0]
        for i in class_in_sample:
            if i != first_class:
                return False

        return True

    def purity_percentage(self, data):
        target = data[:, -1]
        class_count = {}
        for classes in target:
            if classes in class_count.keys():
                class_count[classes] += 1
            else:
                class_count[classes] = 1

        print(class_count, max(class_count.values()), sum(class_count.values()))
        print(target)

        percentage = max(class_count.values()) / sum(class_count.values())
        return percentage

    def find_class(self, data):
        target = data[:, -1]
        minimum = 0
        class_count = {}
        for i in target:
            if i in class_count.keys():
                class_count[i] += 1
            else:
                class_count[i] = 1
        for i in class_count:
            if class_count[i] > minimum:
                minimum = class_count[i]
                class_number = i

        return 'class {}'.format(class_number)

    def partitioning(self, data):  # Pass data without target
        partitions = []
        column_elements = []
        for i in range(data.shape[1] - 2):
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
            if data[i, column + 1] > value:
                data_above.append(list(data[i]))
            else:
                data_below.append(list(data[i]))

        return np.array(data_below), np.array(data_above)

    def entropy(self, data):
        class_count = {}
        entropy = 0
        target = data[:, -1]
        for i in range(len(target)):
            if target[i] in class_count.keys():
                class_count[target[i]] += 1
            else:
                class_count[target[i]] = 1

        for key in class_count:
            entropy += ((class_count[key] / len(data)) * -np.log2(class_count[key] / len(data)))

        return entropy

    def total_entropy(self, data_below, data_above):
        total_instances = len(data_above) + len(data_below)
        total_entropy = (len(data_below) / total_instances * self.entropy(data_below)) + (
                    len(data_above) / total_instances * self.entropy(data_above))

        return total_entropy

    def best_partitioning(self, data, partitioning):
        minimum_entropy = 100
        execute = 0
        l = []
        for i in range(len(partitioning)):
            for value in partitioning[i]:
                data_below, data_above = self.split(data, i, value)
                print('data below is {} and da is {}'.format(data_below, data_above))
                if data_below.size != 0 and data_above.size != 0:
                    execute = 1
                    # print("data below is -- {}, data above is -- {}".format(data_below, data_above))
                    # print("entropy is -- {}".format(self.total_entropy(data_below, data_above)))
                    total_entropy = self.total_entropy(data_below, data_above)
                    l.append(total_entropy)

                    if total_entropy < minimum_entropy:
                        minimum_entropy = total_entropy
                        column = i
                        partition_value = value
                        db = data_below
                        da = data_above

        # if execute == 1:
        #     print(minimum_entropy, column, partition_value)
        #     return minimum_entropy, column, partition_value, l, db, da
        # else:
        #     return 0,0,0,0, True, True

        if execute == 1:
            return column, partition_value
        else:
            return -1, -1

    def grow(self, data):
        if self.purity(data):
            class_is = self.find_class(data)
            return class_is
        if self.counter == 6:
            return self.find_class(data)
        else:
            self.counter += 1
            partition = self.partitioning(data)
            print(partition)
            column, partition_value = self.best_partitioning(data, partition)
            if column != -1:
                data_below, data_above = self.split(data, column, partition_value)
            else:
                return self.find_class(data)
            print(column, partition_value)
            question = 'if data of column {} > {} '.format(column + 1, partition_value)
            tree = {question: []}

            print(len(data_below), len(data_above))

            if_greater = self.grow(data_above)
            if_smaller = self.grow(data_below)

            if if_greater == if_smaller:
                tree = if_greater
            else:
                tree[question].append(if_smaller)
                tree[question].append(if_greater)

            return tree

#
# xT = np.array([[2,3,4,5],[2,6,7.5,4],[3,6,9.7,3]])
# yT = np.array([0,0,1])
# obj = DecisionTree(x[:,2:], y)
# par = obj.partitioning((obj.add_indices(np.append(x[:,2:], y.reshape(len(x[:,2:]), 1), axis = 1)))[:,:3])
# db,da = obj.split(obj.add_indices(np.append(x[:,2:], y.reshape(len(x[:,2:]), 1), axis = 1)),2,0.8)
# en = obj.entropy(np.append(xT, yT.reshape(len(xT), 1), axis = 1))
# en2 = obj.total_entropy(db,da)
# m,c,v,l = obj.best_partitioning(np.append(x[:,2:], y.reshape(len(x), 1), axis = 1), par)
#
# o = []
# for i in range(2):
#     for value in par[i]:
#         db, da = obj.split(obj.add_indices(np.append(x[:, 2:], y.reshape(len(x[:,2:]), 1), axis = 1)), i+1, value)
#         if db.size != 0 and da.size != 0:
#             o.append(obj.total_entropy(db, da))
#
#
# b, a = obj.grow(obj.add_indices(np.append(x[:,2:], y.reshape(len(x[:, 2:]), 1), axis = 1)))
#
obj = DecisionTree(x[:, 2:], y)
n = obj.train()
