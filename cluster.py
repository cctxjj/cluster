import pandas as pd
from sklearn.cluster import KMeans

iris_data_train = pd.read_csv("./iris_data_train.csv", sep=";", index_col=0).to_numpy()
iris_labels_train = pd.read_csv("./iris_labels_train.csv", sep=";", index_col=0).to_numpy().flatten()

iris_data_test = pd.read_csv("./iris_data_test.csv", sep=";", index_col=0).to_numpy()
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

        self.accuracy = self.__train__()

    def __train__(self):
        self.cluster.fit(self.traindata)
        prediction = self.cluster.predict(self.testdata)

        expected = {}
        for i in range(0, self.cluster.n_clusters):
            expected[i] = 0
        for item in self.test_result_data:
            expected[item[1]] += 1

        var = 0
        for i in range(0, 3):
            for x in range(0, 3):
                if [*prediction].count(i) == expected[x]:
                    var += 1
                    break

        return var / self.cluster.n_clusters * 100

    def predict(self, data):
        return self.cluster.predict(data)


def predictionTest(ind):
    predictor = Predictor()
    print(predictor.accuracy)
    for index, el in enumerate(iris_data_test):
        if iris_labels_test[index][1] == ind:
            print(predictor.predict([el]))

