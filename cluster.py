import pandas as pd
from sklearn.cluster import KMeans
import numpy

iris_data_train = pd.read_csv("./iris_data_train.csv", sep=";", index_col=0).to_numpy()
iris_labels_train = pd.read_csv("./iris_labels_train.csv", sep=";", index_col=0).to_numpy().flatten()

iris_data_test = pd.read_csv("./iris_data_test.csv", sep=";").to_numpy()
iris_labels_test = pd.read_csv("./iris_labels_test.csv", sep=";").to_numpy()


class Predictor:

    def __init__(self,
                 traindata=iris_data_train,
                 testdata=pd.read_csv("./iris_data_test.csv", sep=";", index_col=0).to_numpy(),
                 test_result_data=iris_labels_test,
                 clusters=3,
                 iterations=50000):

        self.cluster = KMeans()
        self.traindata = traindata
        self.testdata = testdata
        self.test_result_data = test_result_data
        self.cluster.algorithm = "lloyd"

        self.cluster.n_clusters = clusters
        self.cluster.max_iter = iterations
        self.cluster.n_iter_ = 10000

        self.inaccuracy = self.__train__()

    def __train__(self):
        self.cluster.fit(self.traindata)
        prediction = self.cluster.predict(self.testdata)

        expected = []
        for item in self.test_result_data:
            expected.append(item[1])

        var = 0
        for i in range(0, 3):
            for x in range(0, 3):
                if [*prediction.flatten()].count(i) != expected.count(x):
                    if x == 2:
                        var += 1

        return var / len(prediction)

    @staticmethod
    def getVariation(group1: list, group2: list):
        variation = 0
        for item in group1:
            if group2.count(item) == 0:
                variation += 1
        return variation / len(group1)

    def predict(self, data):
        return self.cluster.predict(data)
