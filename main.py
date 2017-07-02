# Import all the dependencies
import pandas as pd
from sklearn import linear_model
import matplotlib.pyplot as plot

# Read data
data = pd.read_fwf('brain_body.txt')
x_values = data[['Brain']]
y_values = data[['Body']]

# Train model using Linear Regression
body_prediction = linear_model.LinearRegression()
body_prediction.fit(x_values, y_values)

# Visualize results
plot.scatter(x_values, y_values)
plot.plot(x_values, body_prediction.predict(x_values))
plot.show()