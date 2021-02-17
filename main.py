import pandas as pd
import seaborn as sns
import numpy as np
from matplotlib import pyplot as plt
from sklearn.datasets import load_boston
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split    

data = load_boston()
boston_data = pd.DataFrame(data.data, columns=data.feature_names)
boston_target = pd.DataFrame(data.target, columns=['MEDV'])


data_train, data_test, label_train, label_test = train_test_split(boston_data, boston_target, test_size=0.33, random_state=42)
regression_model = LinearRegression()
regression_model.fit(data_train, label_train)
predicted_data = regression_model.predict(data_test)

boston_data['MEDV'] = data.target
fig = plt.figure(figsize=[10, 10])
sns.heatmap(data = boston_data.corr(), annot=True)
fig.savefig('heatmap.png')
plt.close('all')
fig = plt.figure(figsize=[10, 10])
sns.regplot(x='LSTAT', y='MEDV', data=boston_data)
fig.savefig('LSTAT.png')
plt.close()
