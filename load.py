from sklearn import datasets, model_selection

iris_data, iris_labels = datasets.load_iris(return_X_y=True, as_frame=True)

iris_data_train, iris_data_test, iris_labels_train, iris_labels_test = model_selection.train_test_split(iris_data,
                                                                                                        iris_labels,
                                                                                                        test_size=0.1,
                                                                                                        shuffle=True)

iris_data_train.to_csv("iris_data_train.csv", sep=";")
iris_data_test.to_csv("iris_data_test.csv", sep=";")
iris_labels_train.to_csv("iris_labels_train.csv", sep=";")
iris_labels_test.to_csv("iris_labels_test.csv", sep=";")
