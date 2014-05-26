"""
Predicting Baseball Salaries
Using the baseball dataset provided, find the best model in predicting 2012 Salaries for each individual player.
"""

import pandas as pd
import numpy as np
from sklearn import linear_model, metrics

b2011 = pd.read_csv('../../../../data/baseball/baseball_training_2011.csv')
b2012 = pd.read_csv('../../../../data/baseball/baseball_test_2012.csv')

train_X = b2011[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']].values
train_y = b2011['salary'].values

# 2nd order polynominal 
train_X2 = train_X * train_X

test_X = b2012[['G', 'AB', 'R', 'H', 'X2B', 'X3B', 'HR', 'RBI', 'SB', 'CS', 'BB', 'SO', 'IBB', 'HBP', 'SH', 'SF']].values
b2012_csv = b2012[['playerID','yearID', 'salary']]

# Attempt 3.2 : Ridge Regression Model with 2nd order polynominal
ridge = linear_model.Ridge(alpha=1)
ridge.fit(train_X2, train_y)

# Checking performance
print 'R-Squared:',ridge.score(train_X2, train_y)
# Checking MSE
print 'MSE:',metrics.mean_squared_error(ridge.predict(train_X2), train_y)
#
# Alpha:  1
# R-Squared: 0.202505296416
# MSE: 1.68054632023e+13

# Outputting to a csv file
print "Outputting submission file as 'submission.csv'"
b2012_csv['predicted'] = ridge.predict(test_X)
b2012_csv.to_csv('submission.csv')

