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
    def __init__(self, total_features: int, num_fragments: int, num_classes: int):
        super(CNN, self).__init__()
        self.num_fragments = num_fragments

        # Calculate the features per fragment
        self.features_per_fragment = (
            total_features - num_fragments - 1
        ) // num_fragments
        assert (
            total_features - num_fragments - 1
        ) % num_fragments == 0, "Total features do not evenly divide into fragments"

        self.conv1 = nn.Conv2d(
            2, 32, kernel_size=3, stride=1, padding=1
        )  # 2 channels: 1 for features, 1 for charges
        self.relu = nn.ReLU()
        self.pool = nn.MaxPool2d(kernel_size=2, stride=2, padding=0)
        self.conv2 = nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1)

        # Calculate the correct size for the fully connected layer input
        self.fc1_input_dim = (
            64 * (self.num_fragments // 4) * (self.features_per_fragment // 4)
        )
        self.fc1 = nn.Linear(self.fc1_input_dim, 128)
        self.fc2 = nn.Linear(128, num_classes)

    def forward(self, x):
        # Remove the last feature
        x = x[:, :-1]

        # Extract charge features (last num_fragments features)
        charges = x[:, -self.num_fragments :]

        # Extract main features (all but last num_fragments features)
        main_features = x[:, : -self.num_fragments]

        # Debugging: Print shapes
        logging.INFO(f"Input shape: {x.shape}")
        logging.INFO(f"Main features shape: {main_features.shape}")
        logging.INFO(f"Charges shape: {charges.shape}")

        # Reshape main features to (batch_size, 1, num_fragments, features_per_fragment)
        batch_size = x.size(0)
        main_features = main_features.view(
            batch_size, 1, self.num_fragments, self.features_per_fragment
        )

        # Reshape charges to match the fragments shape (batch_size, 1, num_fragments, 1)
        charges = charges.view(batch_size, 1, self.num_fragments, 1)

        # Repeat charges to match the feature dimension (batch_size, 1, num_fragments, features_per_fragment)
        charges = charges.repeat(1, 1, 1, self.features_per_fragment)

        # Combine main features and charges along the channel dimension
        x = torch.cat((main_features, charges), dim=1)

        # Debugging: Print the shape after concatenation
        logging.INFO(f"Combined shape: {x.shape}")

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(batch_size, -1)
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
    "CNN",
    "SimpleNN",
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

# None as placeholders for specific classifiers in pytorch
classifiers = [
    None,
    None,
    None,
    None,
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
        num_fragments = len(enzymes[enzyme].splitting_list)
        models = [
            CNN(
                total_features=num_columns - 1,
                num_fragments=num_fragments,
                num_classes=5,
            ),
            SimpleNN(num_classes=unique_count_target, in_features=num_columns - 1),
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
                f1_macro, balanced_acc, auc_score, logloss = train_pytorch_classifier(
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
                all_balanced_accuracies[name_classifier + "_" + enzyme] = balanced_acc
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

        plot_balanced_accuracies(foldername_output, all_balanced_accuracies, enzyme)


if __name__ == "__main__":
    main()
