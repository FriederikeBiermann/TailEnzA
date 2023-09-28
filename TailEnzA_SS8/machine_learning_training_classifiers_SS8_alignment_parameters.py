#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:02:45 2022

@author: friederike
"""

import pandas as pd
import pickle
import numpy as np
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import RandomForestClassifier
from sklearn.ensemble import ExtraTreesClassifier, RandomForestClassifier, AdaBoostClassifier, BaggingClassifier,  VotingClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.ensemble import ExtraTreesClassifier
from test_classification_methods import *
from plot_figures import *
import sys
 


names_classifiers = [
    "ExtraTreesClassifier",
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "BaggingClassifier",
    "DecisionTreeClassifier",
    #"HistGradientBoostingClassifier",
    "MLPClassifier"
]

classifiers = [
    ExtraTreesClassifier(max_depth=15,min_samples_leaf=1,class_weight="balanced"),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(n_estimators=100),
    BaggingClassifier(KNeighborsClassifier(),max_samples=0.5, max_features=0.5),
    DecisionTreeClassifier(max_depth=5),
    #HistGradientBoostingClassifier(max_iter=100),
    MLPClassifier(solver='lbfgs', alpha=1e-5,hidden_layer_sizes=(5, 2), random_state=1)

]

#define max depth of decision tree and other hyperparameters
test_size=0.5
maxd=15
# fill in names of files here!
# generate feature matrix with feature_generation.py first
directory_feature_matrices="/projects/p450/2dpred/TailEnzA/tailenza_noemi/Classifiers/feature_matrices_different_alignments/"
foldername_output="/projects/p450/2dpred/TailEnzA/tailenza_noemi/Classifiers/feature_matrices_different_alignments/"
alignment_parameters=["gomin1_gemin1","gomin1_gemin2","gomin1_gemin5","gomin2_gemin1","gomin2_gemin1_cenmin2","gomin2_gemin2","gomin3_gemin1","gomin3_gemin2","gomin3_gemin5","gomin5_gemin2","gomin8_gemin1","gomin8_gemin2","gomin2_gemin1_cenmin1","gomin2_gemin1_cenmin5"]
enzymes=["ycao","SAM","Methyl","p450"]
#enzymes=["SAM"]
BGC_types=["ripp","nrp","pk"]
# go through all enzymes, split between test/training set and train classifiers on them
for enzyme in enzymes:
    for alignment_parameter in alignment_parameters:
        path_feature_matrix=directory_feature_matrices+enzyme+"_"+alignment_parameter+"_complete_feature_matrix.csv"
        x_train, x_test, y_train, y_test, x_data, y_data= create_training_test_set(path_feature_matrix, test_size)
        for classifier,name_classifier in zip ( classifiers, names_classifiers):
            train_classifier_and_get_accuracies(classifier,name_classifier, enzyme, alignment_parameter, x_data,y_data,x_train,y_train,x_test,y_test, foldername_output)



