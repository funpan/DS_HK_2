# Classwork 05
'''
 1. Go through the same steps, but this time generate a new model use the log of brain and body, 
 which we know generated a much better distribution and cleaner set of data. 
 Compare the results to the original model. Remember that exp() can be used to "normalize" 
 our "logged" values. 
'''

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# Set some Pandas options
pd.set_option('max_columns', 30)
pd.set_option('max_rows', 20)

# Store data in a consistent place
DATA_DIR = '../../../data/'

mammals = pd.read_csv(DATA_DIR + 'mammals.csv')

from sklearn import linear_model

# Make the model object
regr = linear_model.LinearRegression()

# Fit the data
from numpy import log
from numpy import exp

mammals['log_body'] = log(mammals['body'])
mammals['log_brain'] = log(mammals['brain'])

log_body = [[x] for x in mammals['log_body'].values]
log_brain = mammals['log_brain'].values

regr.fit(log_body, log_brain)

# Display the coefficients:
print "Coef of body-brain %f:" % regr.coef_
print "SSE: %f" % np.mean((regr.predict(log_body) - exp(log_brain)) ** 2)
print "Scoring: %f" % regr.score(log_body, log_brain)

plt.scatter(log_body, log_brain)
plt.plot(log_body, regr.predict(log_body), color='blue', linewidth=3)
plt.show()

'''
Conclusion:
Performance for "brain and body" with logged scoring is ~93% accuracy, slightly better than 
without logged scoring ~87%.
'''

####################
'''
2. Using your aggregate data compiled from nytimes1-30.csv, write a python script 
that determines the best model predicting CTR based off of age and gender. 
Since gender is not actually numeric (it is binary), investigate ways to vectorize 
this feature. 
'''
readers = pd.read_csv(DATA_DIR + 'nyagg.csv')
ageGender = readers[["Age","Gender"]].values

# Make the model object
regr = linear_model.LinearRegression()

# Fit the data
ag = ageGender
ctr = readers['Ctr'].values

regr.fit(ag, ctr)

print "Coef of Age-Gender: %s" % regr.coef_
print "SSE: %f" % np.mean((regr.predict(ag) - ctr) ** 2)
print "Scoring: %f" % regr.score(ag, ctr)

####################
'''
3. Compare this practice to making two separate models based on Gender, with Age 
as your one feature predicting CTR. How are your results different? 
Which results would you be more confident in presenting to your manager? Why's that?
'''
gender = [[x] for x in readers['Gender'].values]
age = [[x] for x in readers['Age'].values]

# Make the model object of Gender
regr_gender = linear_model.LinearRegression()
regr_gender.fit(gender, ctr)

print "Coef of Gender: %s" % regr_gender.coef_
print "SSE: %f" % np.mean((regr_gender.predict(gender) - ctr) ** 2)
print "Scoring: %f" % regr_gender.score(gender, ctr)

# Make the model object of Age
regr_age = linear_model.LinearRegression()
regr_age.fit(age, ctr)

print "Coef of Age: %s" % regr_age.coef_
print "SSE: %f" % np.mean((regr_age.predict(age) - ctr) ** 2)
print "Scoring: %f" % regr_age.score(age, ctr)





