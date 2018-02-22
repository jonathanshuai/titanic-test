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

# Find columns that have null values
train_df.isnull().any(axis=0)

# Only 2 null Embarked values... let's drop it
train_df['Embarked'].isnull().sum()
train_df = train_df[train_df['Embarked'].notnull()]

# Fill in missing values with linear interpolation
train_df['Age'] = train_df['Age'].interpolate()
test_df['Age'] = test_df['Age'].interpolate()
test_df['Fare'] = test_df['Fare'].interpolate()

le = LabelEncoder()
le.fit(['A', 'B', 'C', 'D'])

train_df['Pclass'] = le.inverse_transform(train_df['Pclass'])
test_df['Pclass'] = le.inverse_transform(test_df['Pclass'])

from xgboost import XGBClassifier
from sklearn.linear_model import LogisticRegression

# predictor_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Fare', 'Embarked', 'Age']
predictor_columns = ['Pclass', 'Sex', 'SibSp', 'Parch', 'Embarked', 'Age']
label_column = 'Survived'

X_train = pd.get_dummies(train_df[predictor_columns])
y_train = train_df[label_column]

X_test = pd.get_dummies(test_df[predictor_columns])


# Use cross validation grid search to find good parameters
# parameters_list = {'max_depth': [3, 5, 7, 9], 'n_estimators': [50, 100, 150, 200], 'gamma': [0.1, 1, 10, 100, 1000]}
parameters_list = {'max_depth': [3], 'n_estimators': [50], 'gamma': [1]}

clf = GridSearchCV(XGBClassifier(), parameters_list)

#clf = LogisticRegression()

clf.fit(X_train, y_train)

training_predictions = clf.predict(X_train)
testing_predictions = clf.predict(X_test)

training_accuracy = accuracy_score(y_train, training_predictions)

print(training_accuracy)

test_df['Survived'] = testing_predictions
test_df[['PassengerId', 'Survived']].to_csv('predictions', index=False)


