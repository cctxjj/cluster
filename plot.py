import pandas as pd
import matplotlib.pyplot as plt
import numpy as np

iris_data_train = pd.read_csv(".\\iris_data_train.csv", sep=";", index_col=0)
iris_labels_train = pd.read_csv(".\\iris_labels_train.csv", sep=";", index_col=0).to_numpy().flatten()
iris_data_test = pd.read_csv(".\\iris_data_test.csv", sep=";", index_col=0)

x = iris_data_train["sepal length (cm)"].to_numpy()
y = iris_data_train["sepal width (cm)"].to_numpy()

for label in np.unique(iris_labels_train):
    plt.scatter(x[iris_labels_train == label], y[iris_labels_train == label], label=label)

plt.scatter(iris_data_test["sepal length (cm)"], iris_data_test["sepal width (cm)"], c="black", label="?")

plt.xlabel("sepal length (cm)")
plt.ylabel("sepal width (cm)")
plt.legend()
plt.show()
