# -*- coding: utf-8 -*-
"""
Created on Thu Feb 02 09:36:02 2017

@author: JohannesM
"""

import pandas as pd
import matplotlib.pyplot as plt
import os

path = os.path.dirname(__file__)
cancer_table = pd.read_csv( path + "/data.csv" )

#getting an idea of the table
print cancer_table.head(3)
print cancer_table.count()


#cleaning data of unwanted unnamed column (in this step NaN values would be ommitted)
cancer_table.drop('Unnamed: 32', axis=1, inplace=True)
cancer_table.dropna(inplace=True)

#see if data is biased
marginal_count =  len(cancer_table[cancer_table['diagnosis']=='M'].index)
benign_count =  len(cancer_table[cancer_table['diagnosis']=='B'].index)

print marginal_count/float((marginal_count + benign_count)) 

#print scatterplot for finding correlated features
f, (ax1, ax2) = plt.subplots(1, 2)
f.set_figheight(5)
f.set_figwidth(10)
ax1.scatter(cancer_table['radius_mean'],cancer_table['perimeter_mean'],marker='o', c=cancer_table['diagnosis'])
ax1.set_title("radius_men vs. perimeter_mean")
ax2.scatter(cancer_table['radius_mean'],cancer_table['area_mean'],marker='o', c=cancer_table['diagnosis'])
ax2.set_title("radius_mean vs. area_mean")

#Large feature space and only about 550 data points -> Drop spread and worst values and only work with mean values.
#In addition drop strongly correlated values (perimeter_mean, area_mean)
#Idea to improve this: drop highly correlated features and then calculate pca 
cancer_table.drop('area_mean', axis=1, inplace=True)
cancer_table.drop(cancer_table.columns.values[11:31], axis=1, inplace=True)

print cancer_table.head(3)