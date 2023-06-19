# Importing the Dependencies
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score

# Data Collection and Preprocessing
## loading the dataset into a DataFrame
sonar_data = pd.read_csv(r'C:\Users\MAREH WILLIAM\Desktop\UG\SEM 1_300\DCIT 311 - Machine Learning\datasets\Copy of sonar data.csv' , header=None)

## printing the data head
sonar_data.head()

## statistical measures of the data
sonar_data.describe()

## value counts of resultant values
sonar_data[60].value_counts()
## M - Mine
## R - Rock

## Finding the mean of M and R in each feature
sonar_data.groupby(60).mean()

## separating data and labels
X = sonar_data.drop(columns=60, axis =1)
y = sonar_data[60]

# Training and Test Data

## dividing the data into test and train
x_train, x_test, y_train, y_test = train_test_split(X, y, test_size=0.2, stratify=y, random_state=42)
print(X.shape, x_train.shape, x_test.shape)

# Training of Model

model = LogisticRegression()

##training with train data
model.fit(x_train, y_train)

# Model Evaluation

## accuracy on training data
x_train_predict = model.predict(x_train)
training_data_accuracy = accuracy_score(x_train_predict, y_train)
print('Accuracy_Training Data:', training_data_accuracy)

## accuracy on testing data
x_test_predict = model.predict(x_test)
test_data_accuracy = accuracy_score(x_test_predict, y_test)
print('Accuracy_Testing Data:', test_data_accuracy)

# Making a Predictive System
input_data = (0.0116,0.0744,0.0367,0.0225,0.0076,0.0545,0.1110,0.1069,0.1708,0.2271,0.3171,0.2882,0.2657,0.2307,0.1889,0.1791,0.2298,0.3715,0.6223,0.7260,0.7934,0.8045,0.8067,0.9173,0.9327,0.9562,1.0000,0.9818,0.8684,0.6381,0.3997,0.3242,0.2835,0.2413,0.2321,0.1260,0.0693,0.0701,0.1439,0.1475,0.0438,0.0469,0.1476,0.1742,0.1555,0.1651,0.1181,0.0720,0.0321,0.0056,0.0202,0.0141,0.0103,0.0100,0.0034,0.0026,0.0037,0.0044,0.0057,0.0035)

## changing the data type to a np array
input_data_np = np.asarray(input_data)

## reshaping input_data_np as prediction for an instance takes place
input_data_np_reshaped = input_data_np.reshape(1, -1)

prediction = model.predict(input_data_np_reshaped)
print(prediction)

if (prediction[0] == 'R'):
    print('The object is a Rock')
else:
    print('The object is a mine')