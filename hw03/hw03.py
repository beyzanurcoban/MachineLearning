import math
import matplotlib.pyplot as plt
import numpy as np

# read data into memory
data_set_train = np.genfromtxt("hw03_data_set.csv", delimiter=",", skip_footer=122, skip_header=1)
data_set_test = np.genfromtxt("hw03_data_set.csv", delimiter=",", skip_header=151)

# get x and y values
x_train = data_set_train[:, 0]
y_train = data_set_train[:, 1].astype(int)
x_test = data_set_test[:, 0]
y_test = data_set_test[:, 1].astype(int)

# get number of samples
N_train = x_train.shape[0]
N_test = x_test.shape[0]
print("Number of samples (training):", N_train)
print("Number of samples (test):", N_test)

# defining the interval that we want to draw our plots
minimum_value = 1.5
maximum_value = 5.2
data_interval = np.linspace(minimum_value, maximum_value, 801)

# REGRESSOGRAM #
bin_width = 0.37
left_borders = np.arange(minimum_value, maximum_value, bin_width)
right_borders = np.arange(minimum_value + bin_width, maximum_value + bin_width, bin_width)

p_hat1 = np.asarray([(np.sum(y_train[(left_borders[b] < x_train) & (x_train <= right_borders[b])]) / np.sum(
    (left_borders[b] < x_train) & (x_train <= right_borders[b]))) for b in range(len(left_borders))])

plt.figure(figsize=(10, 4))
plt.axis([1.4, 5.5, 40, 100])
plt.title("Regressogram, h = 0.37")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.plot(x_train, y_train, "b.", markersize=6, label="training")
plt.plot(x_test, y_test, "r.", markersize=6, label="test")
plt.legend(loc=2)

for b in range(len(left_borders)):
    plt.plot([left_borders[b], right_borders[b]], [p_hat1[b], p_hat1[b]], "k-")
for b in range(len(left_borders) - 1):
    plt.plot([right_borders[b], right_borders[b]], [p_hat1[b], p_hat1[b + 1]], "k-")
plt.show()

# RMSE FOR REGRESSOGRAM #
rmse1 = 0
for b in range(len(left_borders)):
    summation = np.sum((y_test[(left_borders[b] < x_test) & (x_test <= right_borders[b])] - p_hat1[b]) ** 2)
    rmse1 = rmse1 + summation

rmse1 = np.sqrt(rmse1 / N_test)
print("Regressogram => RMSE is", round(rmse1, 4), "when h is 0.37")

# RUNNING MEAN SMOOTHER #
p_hat2 = np.asarray([np.sum(y_train[((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))]) / np.sum(
    ((x - 0.5 * bin_width) < x_train) & (x_train <= (x + 0.5 * bin_width))) for x in data_interval])

plt.figure(figsize=(10, 4))
plt.axis([1.4, 5.5, 40, 100])
plt.title("Running mean smoother, h = 0.37")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.plot(x_train, y_train, "b.", markersize=6, label="training")
plt.plot(x_test, y_test, "r.", markersize=6, label="test")
plt.plot(data_interval, p_hat2, "k-")
plt.legend(loc=2)
plt.show()

# RMSE FOR RUNNING MEAN SMOOTHER #
rmse2 = 0
for i in range(N_test):
    score2 = np.sum(
        y_train[((x_train - 0.5 * bin_width) < x_test[i]) & (x_test[i] <= (x_train + 0.5 * bin_width))]) / np.sum(
        ((x_train - 0.5 * bin_width) < x_test[i]) & (x_test[i] <= (x_train + 0.5 * bin_width)))
    difference2 = (y_test[i] - score2) ** 2
    rmse2 = rmse2 + difference2

rmse2 = np.sqrt(rmse2 / N_test)
print("Running Mean Smoother => RMSE is", round(rmse2, 4), "when h is 0.37")

# KERNEL SMOOTHER #
p_hat3 = np.asarray([np.sum(
    (1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train) ** 2 / bin_width ** 2)) * y_train) / np.sum(
    1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x - x_train) ** 2 / bin_width ** 2)) for x in data_interval])

plt.figure(figsize=(10, 4))
plt.axis([1.4, 5.5, 40, 100])
plt.title("Kernel smoother, h = 0.37")
plt.xlabel("Eruption time (min)")
plt.ylabel("Waiting time to next eruption (min)")
plt.plot(x_train, y_train, "b.", markersize=6, label="training")
plt.plot(x_test, y_test, "r.", markersize=6, label="test")
plt.plot(data_interval, p_hat3, "k-")
plt.legend(loc=2)
plt.show()

# RMSE FOR KERNEL SMOOTHER #
rmse3 = 0
for i in range(N_test):
    score3 = np.sum(
        (1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x_test[i] - x_train) ** 2 / bin_width ** 2)) * y_train) / np.sum(
        1.0 / np.sqrt(2 * math.pi) * np.exp(-0.5 * (x_test[i] - x_train) ** 2 / bin_width ** 2))
    difference3 = (y_test[i] - score3) ** 2
    rmse3 = rmse3 + difference3

rmse3 = np.sqrt(rmse3 / N_test)
print("Kernel Smoother => RMSE is", round(rmse3, 4), "when h is 0.37")
