import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

USAhousing = pd.read_csv('USA_Housing.csv')

# print(USAhousing.head())

# sns.heatmap(USAhousing.corr(numeric_only=True))
# plt.show()

# plt.scatter(USAhousing['Avg. Area Income'],USAhousing['Price'])
# plt.show()

# plt.scatter(USAhousing['Avg. Area House Age'],USAhousing['Price'])
# plt.show()

# plt.scatter(USAhousing['Avg. Area Number of Bedrooms'],USAhousing['Price'])
# plt.show()

# plt.scatter(USAhousing['Area Population'],USAhousing['Price'])
# plt.show()

# sns.pairplot(USAhousing)
# plt.show()

# sns.distplot(USAhousing['Price'])
# plt.show()

X = USAhousing[['Avg. Area Income', 'Avg. Area House Age', 'Avg. Area Number of Rooms',
               'Avg. Area Number of Bedrooms', 'Area Population']]
y = USAhousing['Price']
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.4, random_state=101)

# Linear Regression
lm = LinearRegression()
lm.fit(X_train,y_train)
predictions = lm.predict(X_test)
# plt.scatter(y_test, predictions)
# plt.show()

print('MAE:', mean_absolute_error(y_test, predictions))
print('MSE:', mean_squared_error(y_test, predictions))
print('RMSE:', np.sqrt(mean_squared_error(y_test, predictions)))
print('R-squared:', r2_score(y_test, predictions))
