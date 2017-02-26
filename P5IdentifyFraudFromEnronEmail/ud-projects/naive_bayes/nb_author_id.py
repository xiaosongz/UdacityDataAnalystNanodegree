#!/usr/bin/python

""" 
    This is the code to accompany the Lesson 1 (Naive Bayes) mini-project. 

    Use a Naive Bayes Classifier to identify emails by their authors
    
    authors and labels:
    Sara has label 0
    Chris has label 1
"""
    
import sys
from time import time
sys.path.append("../tools/")
from email_preprocess import preprocess


### features_train and features_test are the features for the training
### and testing datasets, respectively
### labels_train and labels_test are the corresponding item labels
features_train, features_test, labels_train, labels_test = preprocess()




#########################################################
### your code goes here ###
from sklearn.naive_bayes import GaussianNB
#rename NB classifier
clf = GaussianNB()
#training NB classifier
clf.fit(features_train,labels_train)
#prediction using fitted classifier with test set
#timer Start
t0 = time()
pred = clf.predict(features_test)
#timer end
print "training time:", round(time()-t0, 3), "s"

#check the accuracy of this NB classifier
from sklearn.metrics import accuracy_score
accuracy = accuracy_score(pred,labels_test)
print accuracy

#########################################################




