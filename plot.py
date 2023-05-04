import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

iris_data_train = pd.read_csv("./iris_data_train.csv", sep=";", index_col=0)
iris_labels_train = pd.read_csv("./iris_labels_train.csv", sep=";", index_col=0).to_numpy().flatten()
iris_data_test = pd.read_csv("./iris_data_test.csv", sep=";", index_col=0)

feat_1 = "petal length (cm)"
feat_2 = "petal width (cm)"

x = iris_data_train[feat_1].to_numpy()
y = iris_data_train[feat_2].to_numpy()

for label in np.unique(iris_labels_train):
    plt.scatter(x[iris_labels_train == label], y[iris_labels_train == label], label=label)

plt.scatter(iris_data_test[feat_1], iris_data_test[feat_2], c="black", label="?")

plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.legend()
plt.show()
