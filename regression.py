import pandas as pd
import numpy as np
import sklearn
from sklearn import linear_model
# from sklearn.util import shuffle

# Import all of the data
data = pd.read_csv("student-mat.csv", sep=";")
# Reduce the data down to the attributes that we want
data = data[["G1", "G2", "G3", "studytime", "failures", "absences"]]

# Label - What we're trying to get
predict = "G3"

# Return a new data frame that doesn't have G3 in it
x = np.array(data.drop([predict], 1))

# Actual G3 values
y = np.array(data[predict])

# Splitting up 10% of our data into test samples
x_train, x_test, y_train, y_test = sklearn.model_selection.train_test_split(x, y, test_size=0.1)

# Creating a training model
linear = linear_model.LinearRegression()

# Fit the data to find a best fit line; Stores the line in linear
linear.fit(x_train, y_train)

# Returns a value that represents the accuracy of our model
acc = linear.score(x_test, y_test)
print(acc)

print("Co:  \n", linear.coef_)
print("Intercept: \n", linear.intercept_)

# Use the model to make a prediction
predictions = linear.predict(x_test)

for x in range(len(predictions)):
    # Print out the prediction, the input data, and the actual score
    print(predictions[x], x_test[x], y_test[x])
