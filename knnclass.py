import pandas as pd
import numpy as np

import matplotlib.pyplot as plt

from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

from sklearn.metrics import confusion_matrix, f1_score, classification_report
from sklearn.metrics import mean_squared_error

from sklearn.neighbors import KNeighborsClassifier
from sklearn.neighbors import KNeighborsRegressor

from sklearn import datasets
from tensorflow.keras.datasets import fashion_mnist

import session_info

session_info.show()

iris_data = datasets.load_iris()
pd.DataFrame(iris_data.data, columns=iris_data.feature_names)
iris_data.target_names

iris_X_train, iris_Xtest, iris_y_train, iris_ytest = train_test_split(iris_data.data,
                                                                      iris_data.target,
                                                                      test_size=0.2,
                                                                      random_state=20130810)

CPU times: user 71.9 ms, sys: 873 µs, total: 72.8 ms
Wall time: 73.2 ms

learner_knn

sc
iris_scaledXtest = sc.transform(iris_Xtest)
%%time

learner_knn.predict(iris_scaledXtest)

confusion_matrix(iris_ytest,
                 learner_knn.predict(iris_scaledXtest))

print(classification_report(iris_ytest,
                            learner_knn.predict(iris_scaledXtest)))

data_file = "/content/drive/MyDrive/IIITH-SEDS/data/fashion-mnist.csv"

fashion_mnist_data = pd.read_csv(data_file)

(x_train, y_train), (x_test, y_test) = fashion_mnist.load_data()

plt.figure(figsize=(0.75,0.75))
plt.imshow(x_train[0], cmap='gray')
28*28
fashion_mnist_data.shape

fashion_mnist_data.head()
fashion_X, fashion_y = (fashion_mnist_data.drop('label', axis=1), 
                        fashion_mnist_data.label)
fashion_X_train, fashion_Xtest, fashion_y_train, fashion_ytest = train_test_split(fashion_X,
                                                                                  fashion_y,
                                                                                  test_size=0.2,
                                                                                  random_state=20130810)

sc

StandardScaler(copy=True, with_mean=True, with_std=True)
learner_knn

%%time

fashion_scaledXtest = sc.transform(fashion_Xtest)

fashion_ypred = learner_knn.predict(fashion_scaledXtest)

print(classification_report(fashion_ytest,
                            fashion_ypred))
advertising_df = pd.read_csv("https://raw.githubusercontent.com/nguyen-toan/ISLR/master/dataset/Advertising.csv",
                             index_col=0)
advertising_df.head()

advertising_X, advertising_y = (advertising_df.drop('Sales', axis=1),
                                advertising_df.Sales)
advertising_X_train, advertising_Xtest, advertising_y_train, advertising_ytest = train_test_split(advertising_X,
                                                                                                  advertising_y,
                                                                                                  test_size=0.2,
                                                                                                  random_state=20130810)
sc
learner_knn
advertising_scaledXtest = sc.transform(advertising_Xtest)
mean_squared_error(advertising_ytest,
                   learner_knn.predict(advertising_scaledXtest))


