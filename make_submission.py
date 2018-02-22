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
parameters_list = {'max_depth': [3, 5, 7, 9], 'n_estimators': [50, 100, 150, 200], 'gamma': [0.1, 1, 10, 100, 1000]}
# parameters_list = {'max_depth': [3], 'n_estimators': [50], 'gamma': [1]}
train_m_ind = X_train['Sex_male'] == 1
train_f_ind = X_train['Sex_female'] == 1

clf_m = GridSearchCV(XGBClassifier(), parameters_list)
clf_f = GridSearchCV(XGBClassifier(), parameters_list)

clf_m.fit(X_train[train_m_ind], y_train[train_m_ind])
clf_f.fit(X_train[train_f_ind], y_train[train_f_ind])

train_preds_m = clf_m.predict(X_train[train_m_ind])
train_preds_f = clf_f.predict(X_train[train_f_ind])

train_acc_m = accuracy_score(y_train[train_m_ind], train_preds_m)
train_acc_f = accuracy_score(y_train[train_f_ind], train_preds_f)

print(train_acc_m)
print(train_acc_f)

test_m_ind = X_test['Sex_male'] == 1
test_f_ind = X_test['Sex_female'] == 1

test_preds_m = clf_m.predict(X_test[test_m_ind])
test_preds_f = clf_f.predict(X_test[test_f_ind])

test_df.loc[test_m_ind, 'Survived'] = test_preds_m
test_df.loc[test_f_ind, 'Survived'] = test_preds_f
test_df.loc[:, 'Survived'] = test_df['Survived'].astype('int')

test_df[['PassengerId', 'Survived']].to_csv('predictions', index=False)
  