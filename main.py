import pandas as pd
from sklearn.tree import DecisionTreeRegressor
from sklearn import linear_model
import matplotlib.pyplot as plt
from sklearn.metrics import mean_squared_error

# Read data
data = pd.read_fwf('BrainBody.txt')
x_values = data[['Brain']]
y_values = data[['Body']]

# Train model using Decision Tree Regression and Linear Regression

body_prediction = DecisionTreeRegressor(max_depth=100)# Denotes the tree depth
body_prediction2=linear_model.LinearRegression()
body_prediction.fit(x_values, y_values)
body_prediction2.fit(x_values, y_values)
y_predict=body_prediction.predict(x_values)
y_predict2=body_prediction2.predict(x_values)

# Visualize results

plt.plot(x_values, y_predict)

#Calculate mean squared error using sklearn.metrics
print(mean_squared_error(y_values,y_predict))
print(mean_squared_error(y_values,y_predict2))

