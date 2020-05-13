import numpy as np
from sklearn.datasets import load_breast_cancer, load_iris
import matplotlib.pyplot as plt

dataset = load_iris()
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


class FeatureScaling:
    def __init__(self, data, type='Normalization'):
        self.data = data
        self.type = type

    def Scaling(self):

        if self.type == 'Normalization':
            for i in range(0, self.data.shape[1]):
                self.data[:, i] = (self.data[:, i] - min(self.data[:, i])) / (
                        max(self.data[:, i]) - min(self.data[:, i]))
            return self.data

        elif self.type == 'Standardization':
            for i in range(0, self.data.shape[1]):
                self.data[:, i] = (self.data[:, i] - np.mean(self.data[:, i])) / (np.var(self.data[:, i]))
            return self.data

        elif self.type == 'Mean_Normalization':
            for i in range(0, self.data.shape[1]):
                self.data[:, i] = (self.data[:, i] - np.mean(self.data[:, i])) / (
                        max(self.data[:, i]) - min(self.data[:, i]))
            return self.data


class DecisionTree:
    def __init__(self, training_data, target, impurity_measure='entropy'):
        self.data = self.add_indices(training_data)
        self.__tree = None
        self.impurity_measure = impurity_measure
        self.target = target
        self.data = np.append(self.data, target.reshape(len(self.data), 1), axis=1)
        print(self.data)
        print(self.purity(self.data))
        self.counter = 0

    def train(self):
        self.__tree = self.grow(self.data)
        return self.__tree

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

        return class_number

    def partitioning(self, data):  # Pass data without target
        partitions = []
        column_elements = []
        for i in range(data.shape[1]-1):
            for j in data[:, i]:
                if j in column_elements:
                    continue
                else:
                    column_elements.append(j)
            ps = []
            for i in range(len(column_elements)):
                if i != 0:
                    part = column_elements[i] + column_elements[i - 1]
                    ps.append(part / 2)
            partitions.append(ps)
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
            return data[0][-1]
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
            question = '{} > {}'.format(column + 1, partition_value)
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

    @staticmethod
    def get_condition(str):
        col_num = ""; op = ""; val = ""; space = 0
        for i in str:
            if i == " ": space += 1
            elif space == 0:col_num += i
            elif space == 1: op += i
            elif space == 2: val += i
        return int(col_num)-1, op, float(val)

    def predict(self, instance):
        dict = self.__tree
        while(True):
            key = list(dict.keys()).pop()
            col_num, op, val = self.get_condition(key)
            if op == ">":
                if instance[col_num] > val:
                    if type(dict[key][0]) == type({}): dict = dict[key][0]
                    else: return dict[key][0]
                else:
                    if type(dict[key][1]) == type({}): dict = dict[key][1]
                    else: return dict[key][1]
            else:
                if instance[col_num] < val:
                    if type(dict[key][0]) == type({}): dict = dict[key][0]
                    else: return dict[key][0]
                else:
                    if type(dict[key][1]) == type({}): dict = dict[key][1]
                    else: return dict[key][1]

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

# dataset = load_breast_cancer()
# x = dataset.data
# y = dataset.target
# xTrain, yTrain, xTest, yTest = train_test_split(x, y, 0.7)
# dt = DecisionTree(xTrain, yTrain)
# n = dt.train()
# result = []
# for i in xTest:
#     result.append(dt.predict(i))
# c = 0
# for i in range(len(yTest)):
#     if result[i] == yTest[i]: c += 1
#
# print("Accuray is {}".format(c/len(yTest)))


xTrain, yTrain, xTest, yTest = train_test_split(x, y, 0.8)
obj = DecisionTree(xTrain, yTrain)
n = obj.train()
result = []
for i in xTest:
    result.append(obj.predict(i))








