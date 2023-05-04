import pandas as pd
from sklearn.cluster import KMeans
import numpy

iris_data_train = pd.read_csv("./iris_data_train.csv", sep=";", index_col=0).to_numpy()
iris_labels_train = pd.read_csv("./iris_labels_train.csv", sep=";", index_col=0).to_numpy().flatten()

iris_data_test = pd.read_csv("./iris_data_test.csv", sep=";", index_col=0).to_numpy()



def some_function():
    cluster = KMeans(n_clusters = 3)
    cluster.fit(iris_data_train)
    prediction = cluster.predict(iris_data_test)
    return prediction


print(some_function())
