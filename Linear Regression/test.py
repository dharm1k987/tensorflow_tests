import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
from sklearn.utils import shuffle
import matplotlib.pyplot as pyplot
import pickle
from matplotlib import style

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

# run the regression a total of 30 times, and save the data which has the highest accuracy
# in studentModel.pickle
"""
bestScore = 0
for i in range(0, 30):
    # x train and y train are a subset of x and y
    X_train, X_test, y_train, y_test = sklearn.model_selection.train_test_split(X, y, test_size=0.1)

    # create a linear regression model
    linear = linear_model.LinearRegression()

    # fit it to the TRAINING data
    linear.fit(X_train, y_train)

    # compare it against the TESTING data
    accuracy = linear.score(X_test, y_test)

    print("Accuracy was {}".format(accuracy))

    if accuracy > bestScore:
        bestScore = accuracy
        # save model as a pickle file
        with open("studentModel.pickle", "wb") as f:
            pickle.dump(linear, f)
"""

# read in pickle file (which has the highest accuracy)
pickle_in = open("studentModel.pickle", "rb")

linear = pickle.load(pickle_in)

# equation of the plane
print("{}x + {}".format(linear.coef_, linear.intercept_))

# make predictions on the TESTING data
predictions = linear.predict(X_test)

# print prediction, what we predicted on, and the actual result
for x in range(0, len(predictions)):
    print(predictions[x], X_test[x], y_test[x])


######### GRID PRINTING ON DATA #########
style.use("ggplot")

# X attribute
p = "G1"
pyplot.scatter(data[p], data[predict])
pyplot.xlabel(p)
pyplot.ylabel("Final Grade (G3)")
pyplot.show()