import pandas as pd
import numpy as np
from sklearn import tree
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.neural_network import MLPClassifier
import time
import sys
from pdb import set_trace
import re
##################################
sys.path.insert(0,'../..')
import my_evaluation
import my_GA


class my_model():

    def obj_func(self, predictions, actuals, pred_proba=None):
        # One objectives: higher f1 score
        eval = my_evaluation(predictions, actuals, pred_proba)
        return [eval.f1()]

    def fit(self, X, y):
        # clean up
        X = X.fillna("")
        X["text_col"] = X["title"] + " " + X["location"]+ " " + X["description"] + " " + X["requirements"] # concatenating text columns
        X = X.drop(columns = ["title","location","description","requirements"])
        X["text_col"] = X["text_col"].apply(lambda x: re.sub("[^a-zA-Z\s]+", "", x)) # removing special characters
        
        # preprocessing
        self.preprocessor = CountVectorizer(stop_words="english") 
        XX = self.preprocessor.fit_transform(X["text_col"])
        XX = pd.DataFrame(XX.toarray())

        # Neural network MLPClassifier to consolidate text predictions
        self.clf = MLPClassifier(hidden_layer_sizes=(100,), solver="adam")
        self.clf.fit(XX,y)
        y_pred = self.clf.predict(XX)
        
        X["y_pred"] = y_pred # derived column for predictions of MLPClassifier
        X = X.drop(columns="text_col")
        
        # Using Decision Tree classifier for classification
        self.clf2 = tree.DecisionTreeClassifier()
        self.clf2.fit(X, y)
        
        return

    def predict(self, X):
        # Replicating same clean up, preprocessing for testing data
        X = X.fillna("")
        X["text_col"] = X["title"] + " " + X["location"]+ " " + X["description"] + " " + X["requirements"]
        X = X.drop(columns = ["title","location","description","requirements"]) 
        X["text_col"] = X["text_col"].apply(lambda x: re.sub("[^a-zA-Z\s]+", "", x))
        
        XX = self.preprocessor.transform(X["text_col"])
        XX = pd.DataFrame(XX.toarray())

        pred_y = np.round(self.clf.predict(XX))
        
        X["y_pred"] = pred_y
        X = X.drop(columns="text_col")        
        predictions = np.round(self.clf2.predict(X))        
        return predictions
