#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:02:45 2022

@author: friederike
"""

import os
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.neural_network import MLPClassifier
from classification_methods import (
    plot_balanced_accuracies,
    plot_cross_val_scores_with_variance,
    train_classifier_and_get_accuracies,
    create_training_test_set,
    train_pytorch_classifier,
)
from tailenza.data.enzyme_information import BGC_types, enzymes
import logging

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


# Define the Simple Neural Network
class SimpleNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(SimpleNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu2 = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = self.relu(self.fc1(x))
        x = self.relu2(self.fc2(x))
        x = self.fc3(x)
        return x


# Define the Convolutional Neural Network (CNN)
class CNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(CNN, self).__init__()
        self.conv1 = nn.Conv2d(1, 32, kernel_size=3, stride=1, padding=1)
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)
        self.fc1 = nn.Linear(
            64 * ((in_features - 1) // 4) * ((in_features - 1) // 4), 128
        )
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))
        x = x.view(-1, 64 * ((in_features - 1) // 4) * ((in_features - 1) // 4))
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        return x


# Define the Recurrent Neural Network (RNN)
class RNN(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, num_classes: int):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size
        self.rnn = nn.RNN(in_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# Define the Long Short-Term Memory (LSTM)
class LSTM(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, num_classes: int):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(in_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        return out


# Update the list of classifier names and classifiers
names_classifiers = [
    "SimpleNN",
    "CNN",
    "RNN",
    "LSTM",
    "ExtraTreesClassifier",
    "RandomForestClassifier",
    "AdaBoostClassifier",
    "DecisionTreeClassifier",
    "MLPClassifier",
]

unique_count_target = (
    10  # Replace with the actual number of unique classes in your target variable
)
num_columns = 20  # Replace with the actual number of columns in your dataset

classifiers = [
    SimpleNN(num_classes=unique_count_target, in_features=num_columns - 1),
    CNN(num_classes=unique_count_target, in_features=num_columns - 1),
    RNN(in_features=num_columns - 1, hidden_size=20, num_classes=unique_count_target),
    LSTM(in_features=num_columns - 1, hidden_size=20, num_classes=unique_count_target),
    ExtraTreesClassifier(max_depth=15, min_samples_leaf=1, class_weight="balanced"),
    RandomForestClassifier(max_depth=5, n_estimators=10, max_features=1),
    AdaBoostClassifier(n_estimators=100),
    DecisionTreeClassifier(max_depth=5),
    MLPClassifier(
        solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
    ),
]

# Define max depth of decision tree and other hyperparameters
test_size = 0.5
maxd = 15
label_mapping = BGC_types
directory_feature_matrices = "../preprocessing/preprocessed_data/dataset_transformer"
foldername_output = "../classifiers/Test_transformer/"


def main():
    # Go through all enzymes, split between test/training set and train classifiers on them
    for enzyme in enzymes:
        all_cross_validation_scores = {}
        all_balanced_accuracies = {}
        path_feature_matrix = os.path.join(
            directory_feature_matrices, f"{enzyme}_BGC_type_feature_matrix.csv"
        )
        df = pd.read_csv(path_feature_matrix)
        num_columns = df.shape[1]
        unique_count_target = df["target"].nunique()

        models = [
            SimpleNN(num_classes=unique_count_target, in_features=num_columns - 1),
            CNN(num_classes=unique_count_target, in_features=num_columns - 1),
            RNN(
                in_features=num_columns - 1,
                hidden_size=20,
                num_classes=unique_count_target,
            ),
            LSTM(
                in_features=num_columns - 1,
                hidden_size=20,
                num_classes=unique_count_target,
            ),
        ]

        criteria = [nn.CrossEntropyLoss() for _ in range(len(models))]
        optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

        x_train, x_test, y_train, y_test, x_data, y_data = create_training_test_set(
            path_feature_matrix, test_size
        )

        classifiers[0:4] = [
            (model, criterion, optimizer)
            for model, criterion, optimizer in zip(models, criteria, optimizers)
        ]

        for classifier, name_classifier in zip(classifiers, names_classifiers):
            if name_classifier in ["SimpleNN", "CNN", "RNN", "LSTM"]:
                model, criterion, optimizer = classifier
                metrics = train_pytorch_classifier(
                    model,
                    criterion,
                    optimizer,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    name_classifier,
                    enzyme,
                    foldername_output,
                    label_mapping,
                )
            else:
                (
                    all_cross_validation_scores[name_classifier + "_" + enzyme],
                    all_balanced_accuracies[name_classifier + "_" + enzyme],
                ) = train_classifier_and_get_accuracies(
                    classifier,
                    name_classifier,
                    enzyme,
                    x_data,
                    y_data,
                    x_train,
                    y_train,
                    x_test,
                    y_test,
                    foldername_output,
                    BGC_types,
                )

        plot_cross_val_scores_with_variance(
            all_cross_validation_scores, foldername_output, enzyme
        )
        plot_balanced_accuracies(foldername_output, all_balanced_accuracies, enzyme)


if __name__ == "__main__":
    main()
