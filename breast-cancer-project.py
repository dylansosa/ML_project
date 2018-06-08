# @author dylan sosa
# Dr. Scannell

import numpy as np
import pandas as pd
import keras, sys
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import SGD
from keras.callbacks import TensorBoard
from sklearn import svm
from sklearn import metrics
from sklearn.metrics import precision_recall_curve
from sklearn.metrics import average_precision_score
from sklearn.metrics import accuracy_score
from sklearn.ensemble import RandomForestClassifier
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.naive_bayes import GaussianNB
from sklearn.model_selection import RandomizedSearchCV
from sklearn.svm import SVC
from sklearn.pipeline import Pipeline
from sklearn.linear_model import LogisticRegression
from scipy.stats import reciprocal, uniform


np.random.seed(42)
reload(sys)
sys.setdefaultencoding('utf-8')
#################
### LOAD DATA ###
#################
data = pd.read_csv('/Users/dylansosa/Documents/SLU/5.2/Machine Learning/Individual Project/cancer/data.csv', header =0)
data.drop("Unnamed: 32",axis=1,inplace=True)
data.drop("id",axis=1,inplace=True)

#########################
### FEATURE SELECTION ###
#########################
features_mean= list(data.columns[1:11])
features_se= list(data.columns[11:20])
features_worst=list(data.columns[21:31])
features = list(data.columns[:31])
# print features
data['diagnosis']=data['diagnosis'].map({'M':1,'B':0})
# malignant, benign
# print data.describe()
# now lets draw a correlation graph so that we can remove multi colinearity it means the columns are
# dependenig on each other so we should avoid it because what is the use of using same column twice
# lets check the correlation between features
# now we will do this analysis only for features_mean then we will do for others and will see who is doing best
corr = data[features_se].corr() # .corr is used for find corelation
#print corr
prediction_var = ['texture_mean','perimeter_mean','smoothness_mean','compactness_mean','symmetry_mean',
                'texture_se','perimeter_se','smoothness_se','compactness_se','symmetry_se',
               'texture_worst','perimeter_worst','smoothness_worst','compactness_worst','symmetry_mean']

# now these are the variables which will use for prediction


##################
### SPLIT DATA ###
##################
# prediction_var = features
train, test = train_test_split(data, test_size = 0.2)# in this our main data is splitted into train and test
# we can check their dimension
print('Training set size:',train.shape)
print('Testing set size:',test.shape)
#print(data.shape)
dontuse,dev = train_test_split(data, test_size = 0.3)# in this our main data is splitted into train and test
print('Development set size:',dev.shape)

##############
### LABELS ###
##############
X_train = train[prediction_var]# taking the training data input
y_train=train.diagnosis# This is output of our training data
# same we have to do for test

X_dev= dev[prediction_var] # taking test data inputs
y_dev =dev.diagnosis   #output value of test dat

X_test= test[prediction_var] # taking test data inputs
y_test =test.diagnosis   #output value of test dat
# print X_train.shape



######################
### Random Forest  ###
######################

    #The sub-sample size is always the same as the original input sample size
    #but the samples are drawn with replacement if bootstrap=True (default).

# model1=RandomForestClassifier(n_estimators=100)# a simple random forest model
#model1=RandomForestClassifier(n_estimators=100, criterion='entropy') #a simple random forest model
# model2=RandomForestClassifier(n_estimators=500)# a simple random forest model
# model3=RandomForestClassifier(n_estimators=1000)# a simple random forest model
#
#
# model1.fit(X_train, y_train)
#
# prediction=model1.predict(X_test)
# print metrics.accuracy_score(prediction,y_test)
#entropy got 98
#gini got 99

# prediction=model1.predict(X_dev)
# print metrics.accuracy_score(prediction,y_dev)
#95



# model2.fit(X_train, y_train)
#prediction2=model2.predict(X_test)
#print metrics.accuracy_score(prediction2,y_test)
#99
# prediction2=model1.predict(X_dev)
# print metrics.accuracy_score(prediction2,y_dev)
#93
#
# model3.fit(X_train, y_train)
#prediction3=model3.predict(X_test)
#print metrics.accuracy_score(prediction3,y_test)
#99
# prediction3=model3.predict(X_dev)
# print metrics.accuracy_score(prediction3,y_dev)
#95

##############################
### Support Vector Machine ###
##############################

# model4 = svm.SVC()
# model4.fit(X_train,y_train)

# prediction4=model4.predict(X_dev)
# print metrics.accuracy_score(prediction4,y_dev)
#94

# prediction4=model4.predict(X_test)
# print metrics.accuracy_score(prediction4,y_test)
#98

# print(metrics.classification_report(prediction4,y_test))
# print(metrics.confusion_matrix(prediction4,y_test))

#param_distributions = {"gamma": reciprocal(0.001,0.1), "C":uniform(1,10)}
    #search over C and gamma in rbf kernel
# param_distributions = {"gamma": 0.07815467993169875, "C":3.888044829725808}
# svn_clf = SVC(decision_function_shape = 'ovr')
# svn_clf.fit(X_train, y_train)
# y_pred = svn_clf.predict(X_train)
# print accuracy_score(y_train,y_pred)

#98

######
# search for optimal parameters
# rnd_search = RandomizedSearchCV(svn_clf, param_distributions, n_iter=10, verbose =2)
# rnd_search.fit(X_train, y_train)
# rnd_search.best_score_
#shows best score
# rnd_search.best_estimator_
#query the best parameters
#so C and gamma
#C=3.888044829725808, gamma=0.07815467993169875

#####


# y_score = svn_clf.decision_function(X_test)
# precision, recall, _ = precision_recall_curve(y_test, y_score)
#
# average_precision = average_precision_score(y_test, y_score)
# print('Average precision-recall score: {0:0.2f}'.format(
      # average_precision))
# plt.step(recall, precision, color='b', alpha=0.2,
         # where='post')
# plt.fill_between(recall, precision, step='post', alpha=0.2,
                 # color='b')
#
# plt.xlabel('Recall')
# plt.ylabel('Precision')
# plt.ylim([0.0, 1.05])
# plt.xlim([0.0, 1.0])
# plt.title('2-class Precision-Recall curve: AP={0:0.2f}'.format(average_precision))
# plt.show()



######################
### Neural Network ###
######################

# model = Sequential()
# model.add(Dense(64, activation='tanh', input_shape=(15,)))
# model.add(Dense(64, activation='tanh'))
#model.add(Dense(1, activation='sigmoid')) #0% accuracy
#model.add(Dense(1, activation='softmax')) #This got around 30% accuracy
# model.add(Dense(1, activation='sigmoid')) #This got around 80% accuracy
# model.summary()
#model.compile(loss='binary_crossentropy', optimizer=SGD(lr=0.01), metrics=['accuracy']) #83 with sigmoid and sgd
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy']) # 89% accuracy, 98 with 1000 epochs
#hinge and squared hinge loss had 37%
# tb = TensorBoard('/Users/dylansosa/Documents/SLU/5.2/Machine Learning/logs/cancer',histogram_freq=0, write_graph=True, write_images=True)
#model.fit(X_train, y_train, batch_size=128, epochs=1000, verbose=1, validation_data=(X_dev, y_dev), callbacks=[tb])
#98
# model.fit(X_train, y_train, batch_size=128, epochs=1000, verbose=1, validation_data=(X_test, y_test), callbacks=[tb])
#98

###################
### Naive Bayes ###
###################

#fit a Naive Bayes model to the data
# model = GaussianNB()
# model.fit(X_test, y_test)
# print(model)
#ake predictions
# expected = y_test
# predicted = model.predict(X_test)
#summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
# y_pred = model.predict(X_test)
# print accuracy_score(y_test,y_pred)
#.94 accuracy

# model.fit(X_dev, y_dev)
# print(model)
#make predictions
# expected = y_dev
# predicted = model.predict(X_dev)
# predicted = model.predict(X_dev)
#summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))

###########################
### Logistic Regression ###
###########################

#fit a logistic regression model to the data
# model = LogisticRegression()
# model.fit(X_test, y_test)
# print(model)
#make predictions
# expected = y_test
# predicted = model.predict(X_test)
#summarize the fit of the model
# print(metrics.classification_report(expected, predicted))
# print(metrics.confusion_matrix(expected, predicted))
# y_pred = model.predict(X_test)
# print accuracy_score(y_test,y_pred)
