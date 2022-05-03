import numpy as np
import matplotlib.pyplot as plt

# IMPORTING DATA #
# read data into memory
data_set_train = np.genfromtxt("hw04_data_set.csv", delimiter=",", skip_footer=122, skip_header=1)
data_set_test = np.genfromtxt("hw04_data_set.csv", delimiter=",", skip_header=151)

# get x and y values
x_train = data_set_train[:, 0]
y_train = data_set_train[:, 1].astype(int)
x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1].astype(int)

# get number of samples
N_train = x_train.shape[0]
N_test = x_test.shape[0]
P = 25
print("Number of samples (training):", N_train)
print("Number of samples (test):", N_test)


# BINARY TREE CLASS DEFINITION #
class BinaryTree:
    def __init__(self, x_train, y_train, p):
        self.x_values = x_train[np.argsort(x_train)]
        self.y_values = y_train[np.argsort(x_train)]
        self.pruning = p

        if len(self.x_values) <= self.pruning:
            self.split_condition = False
        else:
            self.split_condition = True

        if self.split_condition:
            self.middle_points = (np.unique(self.x_values)[1:] + np.unique(self.x_values)[:-1]) / 2

    # This method split the tree into two pieces: left child and right child. We set the left and right children of the
    # root node. So, we connect our tree nodes with each other.
    def splitting(self):
        if self.split_condition:
            decide = np.vectorize(self.decide)
            impurity = decide(self.middle_points)
            self.boundary = self.middle_points[impurity.argmin()]

            x_val_left = self.x_values[self.x_values <= self.boundary]
            y_val_left = self.y_values[self.x_values <= self.boundary]
            x_val_right = self.x_values[self.x_values > self.boundary]
            y_val_right = self.y_values[self.x_values > self.boundary]

            self.leftChild = BinaryTree(x_val_left, y_val_left, self.pruning)
            self.leftChild.splitting()
            self.rightChild = BinaryTree(x_val_right, y_val_right, self.pruning)
            self.rightChild.splitting()

    # This method decides which data points that reach the node m should go to the left sub-tree, and
    # which data points should go to the right sub-tree.

    # We compare the x values of the data points and the parameter w0. According to this comparison,
    # we sent some of the data points to the left, and some of them to the right.
    def decide(self, w0):
        x_left = self.x_values[self.x_values <= w0]
        x_right = self.x_values[self.x_values > w0]
        y_left = self.y_values[self.x_values <= w0]
        y_right = self.y_values[self.x_values > w0]

        completeLeft = BinaryTree(x_left, y_left, self.pruning)
        completeRight = BinaryTree(x_right, y_right, self.pruning)

        impurityLeft = np.sum(
            (completeLeft.y_values - (np.sum(completeLeft.y_values) / len(completeLeft.x_values))) ** 2)
        impurityRight = np.sum(
            (completeRight.y_values - (np.sum(completeRight.y_values) / len(completeRight.x_values))) ** 2)
        impurity = (impurityLeft + impurityRight) / len(self.x_values)
        return impurity

    # This method provides us a recursive structure, if the new comer data point is belong to the left tree,
    # we continue with the left tree. We stop if we have number of elements less than or equal to the pruning parameter.
    # If we stop, we return the average output values of that node.
    def recursion(self, new_data_point):
        if not self.split_condition:
            return np.sum(self.y_values) / len(self.y_values)

        if new_data_point <= self.boundary:
            return self.leftChild.recursion(new_data_point)
        else:
            return self.rightChild.recursion(new_data_point)


# DEFINING THE REGRESSION TREE AND PLOTTING #
regDecisionTree = BinaryTree(x_train, y_train, P)
regDecisionTree.splitting()

prediction_line = np.linspace(1, 6, 1001)
recursion_func = np.vectorize(regDecisionTree.recursion)
y_hat_train = recursion_func(prediction_line)

plt.figure(figsize=(10, 4))
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.title("P = 25", fontweight="bold")
plt.plot(x_train, y_train, "b.", markersize=8, alpha=0.5, label="training")
plt.plot(x_test, y_test, "r.", markersize=8, alpha=0.5, label="test")
plt.legend(loc=2)
plt.plot(prediction_line, y_hat_train, 'k')
plt.show()

y_hat_test = recursion_func(x_test)
# calculate rmse by using the formula
rmse = np.sqrt(np.sum((y_test - y_hat_test) ** 2) / N_test)
print("RMSE is", round(rmse, 4), "when P is", regDecisionTree.pruning)

# REPEATING THE SAME LEARNING PROCESS FOR VARYING PRUNING PARAMETERS #
p_array = np.arange(5, 51, 5)
rmse_array = []
#print(p_array)

for p_val in p_array:
    testTree = BinaryTree(x_train, y_train, p_val)
    testTree.splitting()
    recursion_func2 = np.vectorize(testTree.recursion)
    y_hat_iterative = recursion_func2(x_test) # predictions for each p value
    rmse_iterative = np.sqrt(np.sum((y_test - y_hat_iterative) ** 2) / N_test) # calculate rmse for each p value
    rmse_array.append(rmse_iterative)

plt.figure(figsize=(10, 4))
plt.xlabel("Pre-pruning size (P)")
plt.ylabel("RMSE")
plt.plot(p_array, rmse_array, 'ko-')
plt.yticks(np.arange(6, 8, .5))
plt.show()
