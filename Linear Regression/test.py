import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle

data = pd.read_csv("student-mat.csv", sep=";")

# select columns we want
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# example
print(data.head())

# want to predict final grade
predict = "G3"

# data frame without predicted
X = np.array(data.drop([predict], 1))

# resulting array
y = np.array(data[predict])

# x train and y train are a subset of x and y
X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

# create a linear regression model
linear = linear_model.LinearRegression()

# fit it to the TRAINING data
linear.fit(X_train, y_train)

# compare it against the TESTING data
accuracy = linear.score(X_test, y_test)

# equation of the plane
print("{}x + {}".format(linear.coef_, linear.intercept_))

# make predictions on the TESTING data
predictions = linear.predict(X_test)

# print prediction, what we predicted on, and the actual result
for x in range(0, len(predictions)):
    print(predictions[x], X_test[x], y_test[x])