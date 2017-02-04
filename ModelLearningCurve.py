# -*- coding: utf-8 -*-
"""
Created on Fri Feb 03 22:07:18 2017

@author: Johannes Maerkle
notes: Adapted regression learning curve from max_depth for decision trees from udacitiy Boston Housing prices project
"""

import matplotlib.pyplot as pl
import numpy as np
import sklearn.learning_curve as curves
from sklearn.cross_validation import ShuffleSplit

def ModelLearningCurveClassification(clf, X, y, fold_number=5):
    """ Calculates the performance of a classification model clf with varying sizes of training data.
        The learning and testing scores for the model are then plotted. """
    pl.figure(2)
    # Create fold_number cross-validation sets for training and testing
    cv = ShuffleSplit(X.shape[0], n_iter = fold_number, test_size = 0.2, random_state = 0)

    # Generate the training set sizes increasing by 50
    train_sizes = np.rint(np.linspace(1, X.shape[0]*0.8 - 1, 9)).astype(int)

    # Create the figure window
    #fig = pl.figure(figsize=(10,7))

    # Calculate the training and testing scores for classifier clf
    sizes, train_scores, test_scores = curves.learning_curve(clf, X, y, \
        cv = cv, train_sizes = train_sizes, scoring = 'accuracy')
        
        # Find the mean and standard deviation for smoothing
    train_std = np.std(train_scores, axis = 1)
    train_mean = np.mean(train_scores, axis = 1)
    test_std = np.std(test_scores, axis = 1)
    test_mean = np.mean(test_scores, axis = 1)

    # Subplot the learning curve 

    pl.plot(sizes, train_mean, 'o-', color = 'r', label = 'Training Score')
    pl.plot(sizes, test_mean, 'o-', color = 'g', label = 'Testing Score')
    pl.fill_between(sizes, train_mean - train_std, \
        train_mean + train_std, alpha = 0.15, color = 'r')
    pl.fill_between(sizes, test_mean - test_std, \
        test_mean + test_std, alpha = 0.15, color = 'g')
        
    # Labels
    pl.title('Learning Curve for Classifier')
    pl.xlabel('Number of Training Points')
    pl.ylabel('Score')
    pl.xlim([0, X.shape[0]*0.8])
    pl.ylim([-0.05, 1.05])
    
    # Visual aesthetics
    pl.legend(loc='lower left', borderaxespad = 0.)
    pl.tight_layout()
    pl.show()