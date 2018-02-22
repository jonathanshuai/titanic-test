import numpy as np
import pandas as pd
import scipy.optimize
import scipy.stats

from sklearn.model_selection import train_test_split, GridSearchCV
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import accuracy_score
from sklearn.metrics import mean_squared_error

train_file = "./data/train.csv"
test_file = "./data/test.csv"

train_df = pd.read_csv(train_file) 
test_df = pd.read_csv(test_file) 


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

predictor_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Age']
#predictor_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Age']
label_column = 'Survived'

X = pd.get_dummies(train_df[predictor_columns])
y = train_df[label_column]

X_train, X_test, y_train, y_test = train_test_split(X, y)

from xgboost import XGBClassifier

# Use cross validation grid search to find good parameters
parameters_list = {'max_depth': [3, 5, 7, 9], 'n_estimators': [50, 100, 150, 200], 'gamma': [0.1, 1, 10, 100, 1000]}

clf = GridSearchCV(XGBClassifier(), parameters_list)
clf.fit(X_train, y_train)

females = X_test['Sex_female'] == 1
males = X_test['Sex_male'] == 1
female_predictions = clf.predict(X_test[females])
male_predictions = clf.predict(X_test[males])

female_accuracy = accuracy_score(female_predictions, y_test[females])
male_accuracy = accuracy_score(male_predictions, y_test[males])
print(female_accuracy)
print(male_accuracy)

