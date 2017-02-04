# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 09:36:02 2017

@author: Johannes Maerkle
"""
import pandas as pd
import matplotlib.pyplot as plt
from sklearn import tree
from sklearn import cross_validation
from sklearn.ensemble import RandomForestClassifier
import os

import ModelLearningCurve as mlc

path = os.path.dirname(__file__)
cancer_table = pd.read_csv( path + "/data.csv" )

#getting an idea of the table
print cancer_table.head(3)
print cancer_table.count()


#Cleaning data of unwanted unnamed column, id and NaN values (which weren't present in the data set)
cancer_table.drop('Unnamed: 32', axis=1, inplace=True)
cancer_table.drop('id', axis=1, inplace=True)
cancer_table.dropna(inplace=True)

#see if remaining data is biased
malignant_count =  len(cancer_table[cancer_table['diagnosis']=='M'].index)
benign_count =  len(cancer_table[cancer_table['diagnosis']=='B'].index)

print malignant_count/float((malignant_count + benign_count)) 

#idea: radius, area, perimeter should be highly correlated features

f, (ax1, ax2) = plt.subplots(1, 2)
f.set_figheight(5)
f.set_figwidth(10)
ax1.scatter(cancer_table['radius_mean'],cancer_table['perimeter_mean'],marker='o', c=cancer_table['diagnosis'])
ax1.set_title("radius_men vs. perimeter_mean")
ax2.scatter(cancer_table['radius_mean'],cancer_table['area_mean'],marker='o', c=cancer_table['diagnosis'])
ax2.set_title("radius_mean vs. area_mean")
plt.show()



#Large feature space and only about 570 data points -> Drop spread and worst values and only work with mean values.
#In addition drop strongly correlated values (perimeter_mean and area_mean contains reduntant information compared with radius)
#Idea to improve this further: calculate pca after dropping reduntant features 

cancer_table.drop(cancer_table.columns.values[11:31], axis=1, inplace=True)
cancer_table.drop('area_mean', axis=1, inplace=True)
cancer_table.drop('perimeter_mean', axis=1, inplace=True)



#split table in features and target
features = cancer_table.drop('diagnosis', axis=1)
target = cancer_table['diagnosis']


#first create simple classifier and examine learning curve 
clf_tree = tree.DecisionTreeClassifier(random_state=1)

mlc.ModelLearningCurveClassification(clf_tree, features, target)

#k-fold cross validation (5 folds in this case) for measuring the classifiers score (accuracy in this case)
crossValScore=cross_validation.cross_val_score(clf_tree, features, target, cv=5)
print("Accuracy simple decision tree: %0.2f (+/- %0.2f)" % (crossValScore.mean(), crossValScore.std() * 2))

#More sophisticated: random forest
clf_forest = RandomForestClassifier(n_estimators=20, random_state=1)
mlc.ModelLearningCurveClassification(clf_forest, features, target)

#k-fold cross validation (5 folds in this case) for measuring the classifiers score (accuracy in this case)
crossValScore=cross_validation.cross_val_score(clf_forest, features, target, cv=5)
print("Accuracy random forest: %0.2f (+/- %0.2f)" % (crossValScore.mean(), crossValScore.std() * 2))

clf_forest.fit(features,target)
#Advantage in random forests: one gets feature importance for free
featimp = pd.Series(clf_forest.feature_importances_, index=features.columns.values).sort_values(ascending=False)
print(featimp)