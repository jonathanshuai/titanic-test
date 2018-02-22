# Let's do some quick exploratory analysis

import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

import matplotlib as mpl
from matplotlib import pyplot as plt

train_file = "./data/train.csv"
test_file = "./data/test.csv"

train_df = pd.read_csv(train_file) 
test_df = pd.read_csv(test_file) 

# Let's plot some things...
color = train_df['Survived']
x = train_df['Sex']
y = train_df['Age']

fig, ax = plt.subplots()

ax.scatter(x, y, c=color)
ax.legend()
plt.show()

# Let's plot some things...
color = train_df['Survived']
x = train_df['Pclass']
y = train_df['SibSp'] + train_df['Parch']

fig, ax = plt.subplots()

ax.scatter(x, y, c=color)
ax.legend()
plt.show()

# This tells us which columns have null values
train_df.isnull().any(axis=0)

# Only 2 null Embarked values... let's drop it
train_df['Embarked'].isnull().sum()
train_df = train_df[train_df['Embarked'].notnull()]

le = LabelEncoder()
le.fit(['A', 'B', 'C', 'D'])

train_df['Pclass'] = le.inverse_transform(train_df['Pclass'])

# Fill in missing values for age using linear interpolation
train_df['Age'] = train_df['Age'].interpolate()

from sklearn.ensemble import GradientBoostingClassifier
from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC

predictor_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Age']
label_column = 'Survived'

X = pd.get_dummies(train_df[predictor_columns])
y = train_df[label_column]

X_train, X_test, y_train, y_test = train_test_split(X, y)


for clf in [XGBClassifier(), LogisticRegression(), SVC(), GradientBoostingClassifier(n_estimators=50)]:
  print("------{}------".format(clf))
  clf.fit(X_train, y_train)

  training_predictions = clf.predict(X_train)
  testing_predictions = clf.predict(X_test)

  training_accuracy = accuracy_score(y_train, training_predictions)
  testing_accuracy = accuracy_score(y_test, testing_predictions)

  print(training_accuracy)
  print(testing_accuracy)

