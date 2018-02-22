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


# NN stuff
X_train = X_train.as_matrix()
X_test = X_test.as_matrix()

y_train = y_train.values
y_test = y_test.values

import torch
import torch.autograd as autograd
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

class NNClassifier(nn.Module):

  def __init__(self, n_input, n_hidden, n_labels):
    super(NNClassifier, self).__init__()
    self.i2h = nn.Linear(n_input, n_hidden)
    self.h2h = nn.Linear(n_hidden, n_hidden)
    self.h2o = nn.Linear(n_hidden, n_labels)

  def forward(self, input_vec):
    hidden = self.i2h(input_vec)
    output = self.h2o(hidden)
    output = self.h2h(hidden)
    log_probs = F.log_softmax(output, dim=0)
    return log_probs

# Train the model
N_EPOCHS = 400
n_train_samples = X_train.shape[0]

loss_function = nn.NLLLoss()
model = NNClassifier(X_train.shape[1], 16, 2)
optimizer = optim.SGD(model.parameters(), lr=0.00009, weight_decay=1e-3, momentum=0.9)

# For n epochs...
for epoch in range(N_EPOCHS):
  total_loss = torch.Tensor([0])

  random_indices = np.random.permutation(n_train_samples)

  for index in random_indices:
    sample = X_train[index]
    label = int(y_train[index])

    # Initialize hidden layer
    sample_vector = autograd.Variable(torch.FloatTensor(sample))

    model.zero_grad()
    output = model(sample_vector).view(1, -1)
    
    loss = loss_function(output, autograd.Variable(torch.LongTensor([label])))
    loss.backward()
    # torch.nn.utils.clip_grad_norm(model.parameters(), MAX_NORM)
    optimizer.step()
    total_loss += loss.data

  print(torch.norm(next(model.parameters()).grad))
  print("[epoch {}] {}".format(epoch, total_loss))


training_predictions = []
for i in range(X_train.shape[0]):
  sample_vector = autograd.Variable(torch.FloatTensor(X_train[i]))
  output = model(sample_vector).view(1, -1)
  training_predictions.append(np.argmax(output.data.numpy()[0]))

training_accuracy = accuracy_score(y_train, training_predictions)

testing_predictions = []
for i in range(X_test.shape[0]):
  sample_vector = autograd.Variable(torch.FloatTensor(X_test[i]))
  output = model(sample_vector).view(1, -1)
  testing_predictions.append(np.argmax(output.data.numpy()[0]))

testing_accuracy = accuracy_score(y_test, testing_predictions)



