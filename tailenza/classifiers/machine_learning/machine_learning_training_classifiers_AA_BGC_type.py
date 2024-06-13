#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Wed Feb 16 18:02:45 2022

@author: friederike
"""

import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.ensemble import (
    ExtraTreesClassifier,
    RandomForestClassifier,
    AdaBoostClassifier,
)
from sklearn.neural_network import MLPClassifier
from tailenza.classifiers.machine_learning.classification_methods import (
    plot_balanced_accuracies,
    plot_cross_val_scores_with_variance,
    train_classifier_and_get_accuracies,
    create_training_test_set,
    train_pytorch_classifier,
)
from tailenza.data.enzyme_information import BGC_types, enzymes
import logging
import argparse

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)



# FFNN Model 1: Basic Feedforward Neural Network
class BasicFFNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(BasicFFNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 64)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


# FFNN Model 2: Intermediate Feedforward Neural Network
class IntermediateFFNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(IntermediateFFNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 128)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(64, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.fc3(x)
        x = F.softmax(x, dim=1)
        return x


# FFNN Model 3: Advanced Feedforward Neural Network
class AdvancedFFNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(AdvancedFFNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 256)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(32, num_classes)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        x = F.relu(self.fc4(x))
        x = self.fc5(x)
        x = F.softmax(x, dim=1)
        return x


class VeryAdvancedFFNN(nn.Module):
    def __init__(self, in_features: int, num_classes: int):
        super(VeryAdvancedFFNN, self).__init__()
        self.fc1 = nn.Linear(in_features, 512)
        self.relu = nn.ReLU()
        self.fc2 = nn.Linear(512, 256)
        self.relu = nn.ReLU()
        self.fc3 = nn.Linear(256, 128)
        self.relu = nn.ReLU()
        self.fc4 = nn.Linear(128, 64)
        self.relu = nn.ReLU()
        self.fc5 = nn.Linear(64, 32)
        self.relu = nn.ReLU()
        self.fc6 = nn.Linear(32, num_classes)

        self.dropout = nn.Dropout(0.5)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        x = F.relu(self.fc2(x))
        x = self.dropout(x)
        x = F.relu(self.fc3(x))
        x = self.dropout(x)
        x = F.relu(self.fc4(x))
        x = F.relu(self.fc5(x))
        x = self.fc6(x)
        x = F.softmax(x, dim=1)
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

        x = self.pool(self.relu(self.conv1(x)))
        x = self.pool(self.relu(self.conv2(x)))

        x = x.view(batch_size, -1)
        x = self.relu(self.fc1(x))
        x = self.fc2(x)
        x = F.softmax(x, dim=1)
        return x


# Define the Recurrent Neural Network (RNN)
class RNN(nn.Module):
    def __init__(
        self, in_features: int, hidden_size: int, num_fragments: int, num_classes: int
    ):
        super(RNN, self).__init__()
        self.hidden_size = hidden_size

        self.num_fragments = num_fragments
        self.features_per_fragment = (
            in_features - num_fragments - 1
        ) // num_fragments + 1
        assert (
            in_features - num_fragments - 1
        ) % num_fragments == 0, "Total features do not evenly divide into fragments"
        self.rnn = nn.RNN(self.features_per_fragment, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Remove the last feature
        x = x[:, :-1]

        # Extract charge features (last num_fragments features)
        charges = x[:, -self.num_fragments :]

        # Extract main features (all but last num_fragments features)
        main_features = x[:, : -self.num_fragments]

        # Reshape main features to (batch_size, num_fragments, features_per_fragment)
        batch_size = x.size(0)
        seq_length = self.num_fragments
        input_size = main_features.size(1) // self.num_fragments
        main_features = main_features.view(batch_size, seq_length, input_size)

        # Reshape charges to match the sequence shape (batch_size, num_fragments, 1)
        charges = charges.view(batch_size, seq_length, 1)

        # Combine main features and charges along the last dimension
        x = torch.cat((main_features, charges), dim=2)

        # Initialize hidden state
        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)

        out, _ = self.rnn(x, h0)
        out = out[:, -1, :]
        out = self.fc(out)
        x = F.softmax(x, dim=1)
        return out


# Define the Long Short-Term Memory (LSTM)
class LSTM(nn.Module):
    def __init__(self, in_features: int, hidden_size: int, num_classes: int):
        super(LSTM, self).__init__()
        self.hidden_size = hidden_size
        self.lstm = nn.LSTM(in_features, hidden_size, batch_first=True)
        self.fc = nn.Linear(hidden_size, num_classes)

    def forward(self, x):
        # Ensure input has three dimensions
        if len(x.shape) == 2:
            x = x.unsqueeze(1)  # Add a sequence length dimension
        elif len(x.shape) != 3:
            raise RuntimeError(f"Input must have 3 dimensions, got {x.shape}")

        h0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        c0 = torch.zeros(1, x.size(0), self.hidden_size).to(x.device)
        out, _ = self.lstm(x, (h0, c0))
        out = out[:, -1, :]
        out = self.fc(out)
        out = F.softmax(out, dim=1)
        return out



def main():
    # Update the list of classifier names and classifiers
    names_classifiers = [
        # "RNN",
        "ExtraTreesClassifier",
        #"CNN",
        "LSTM",
        "BasicFFNN",
        "IntermediateFFNN",
        "AdvancedFFNN",
        "VeryAdvancedFFNN",
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
        ExtraTreesClassifier(max_depth=25, min_samples_leaf=1, class_weight="balanced"),
        #None,
        None,
        None,
        None,
        None,
        None,
    
        RandomForestClassifier(max_depth=25, n_estimators=10, max_features=1),
        AdaBoostClassifier(n_estimators=100),
        DecisionTreeClassifier(max_depth=25),
        MLPClassifier(
         solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
        ),
    ]
    
    # Define max depth of decision tree and other hyperparameters
    test_size = 0.5
    maxd = 15

    if MODE == "BGC":
        label_mapping = BGC_types
        directory_feature_matrices = (
            "../preprocessing/preprocessed_data/dataset_transformer_new"
        )
        foldername_output = "../classifiers/Test_transformer_BGC_type/"
    elif MODE == "metabolism":
        label_mapping = ["primary_metabolism", "secondary_metabolism"]
        directory_feature_matrices = (
         "../preprocessing/preprocessed_data/dataset_transformer_without_divergent"
        )
        foldername_output = "../classifiers/Test_transformer_metabolism/"
    else:
        raise ValueError (f"MODE must be BGC or metabolism, not {MODE}")

    argparser = argparse.ArgumentParser()
    argparser.add_argument("--mode", type=str, default="BGC")
    argparser.add_argument("--device", type=str, default="cuda")

    args = argparser.parse_args()
    MODE = args.mode

    # Check if GPU is available
    device = torch.device(args.device if torch.cuda.is_available() else "cpu")

    all_metrics = []
    # Go through all enzymes, split between test/training set and train classifiers on them
    for enzyme in enzymes:
        if enzyme == "P450":
            continue
        # ycao BGC classification makes no sense because all ycaos in antismash DB are RiPPs after filtering 
        if enzyme == "ycao" and MODE == "BGC":
            continue
        all_cross_validation_scores = {}
        all_balanced_accuracies = {}
        path_feature_matrix = os.path.join(
            directory_feature_matrices, f"{enzyme}_{MODE}_type_feature_matrix.csv"
        )
        df = pd.read_csv(path_feature_matrix)
        rows_with_nan = df[df.isna().any(axis=1)]
        df.replace([np.inf, -np.inf], np.nan, inplace=True)
        columns_with_nan = df.columns[df.isna().any()].tolist()
        #logging.debug(f"rows with nan: {rows_with_nan[['f2_0','sbr4_932', 'bridging_413']]}")
        logging.debug(f"colums with nan: {columns_with_nan}")

        initial_row_count = df.shape[0]
        df = df.dropna()
        final_row_count = df.shape[0]
        removed_row_count = initial_row_count - final_row_count
        logging.info(f"Dropped {removed_row_count} lines with nan values")
        num_columns = df.shape[1]
        logging.info(f"Number of colums: {num_columns}")
        unique_count_target = df["target"].nunique()
        num_fragments = len(enzymes[enzyme]["splitting_list"])

        models = [
            # RNN(
            #     in_features=num_columns - 1,
            #     hidden_size=20,
            #     num_classes=unique_count_target,
            #     num_fragments=num_fragments,
            # ),
           
            #CNN(
            #    total_features=num_columns - 1,
            #    num_fragments=num_fragments,
            #    num_classes=unique_count_target,
            #).to(device),
            LSTM(
                in_features=num_columns - 1,
                hidden_size=20,
                num_classes=unique_count_target,
            ).to(device),
            BasicFFNN(num_classes=unique_count_target, in_features=num_columns - 1).to(
                device
            ),
            IntermediateFFNN(
                num_classes=unique_count_target, in_features=num_columns - 1
            ).to(device),
            AdvancedFFNN(
                num_classes=unique_count_target, in_features=num_columns - 1
            ).to(device),
            VeryAdvancedFFNN(
                num_classes=unique_count_target, in_features=num_columns - 1
            ).to(device),
        ]

        criteria = [nn.CrossEntropyLoss() for _ in range(len(models))]
        optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

        x_train, x_test, y_train, y_test, x_data, y_data = create_training_test_set(
            path_feature_matrix, test_size
        )

        
        classifiers[1:6] = [
            (model, criterion, optimizer)
            for model, criterion, optimizer in zip(models, criteria, optimizers)
        ]

        for classifier, name_classifier in zip(classifiers, names_classifiers):
            if name_classifier in [
                "SimpleNN",
                "CNN",
                "RNN",
                "LSTM",
                "BasicFFNN",
                "IntermediateFFNN",
                "AdvancedFFNN",
                "VeryAdvancedFFNN",
            ]:
                model, criterion, optimizer = classifier
                f1_macro, balanced_acc, auc_score, logloss, metrics = (
                    train_pytorch_classifier(
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
                )
                all_metrics.append(metrics)
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
                    label_mapping,
                )

        plot_balanced_accuracies(foldername_output, all_balanced_accuracies, enzyme)
    metrics_df = pd.concat(all_metrics, axis=0)
    date_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
    metrics_df.to_csv(
        os.path.join(foldername_output, f"metrics_all_classifiers{date_time}.csv"),
        index=False,
    )


if __name__ == "__main__":
   main()
