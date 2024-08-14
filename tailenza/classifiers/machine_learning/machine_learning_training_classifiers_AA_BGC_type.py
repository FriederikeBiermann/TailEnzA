import os
import time
from datetime import datetime
import pandas as pd
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.optim.lr_scheduler import ReduceLROnPlateau
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
    TrainerPytorch,
    TrainerScikitLearn,
    create_training_test_set,
)
from tailenza.classifiers.machine_learning.classifiers import (
    LSTM,
    VeryAdvancedFFNN,
    AdvancedFFNN,
    IntermediateFFNN,
    BasicFFNN,
)
from tailenza.data.enzyme_information import BGC_types, enzymes
import logging
import argparse
import matplotlib.pyplot as plt
from typing import Dict, List, Tuple

logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)


def plot_balanced_accuracies(
    foldernameoutput: str, all_balanced_accuracies: Dict[str, float], enzyme: str
) -> None:
    """
    Plots and saves the balanced accuracies of different classifiers.

    Args:
        foldernameoutput (str): The directory where the output plot will be saved.
        all_balanced_accuracies (Dict[str, float]): A dictionary with classifier names as keys and their balanced accuracy scores as values.
        enzyme (str): The name of the enzyme or dataset being analyzed.

    Returns:
        None
    """
    try:
        logging.info("Starting plot creation for balanced accuracies.")

        labels = list(all_balanced_accuracies.keys())
        scores = list(all_balanced_accuracies.values())
        logging.debug(f"Classifier labels: {labels}")
        logging.debug(f"Balanced accuracy scores: {scores}")

        plt.figure(figsize=(14, 10))
        plt.bar(labels, scores, align="center", alpha=0.7)
        plt.ylabel("Balanced Accuracy Score")
        plt.title(f"Balanced Accuracy of Different Classifiers for {enzyme}")
        plt.xticks(rotation=45)
        plt.tight_layout()

        # Save the figure
        output_path = f"{foldernameoutput}{enzyme}_balanced_accuracies.png"
        plt.savefig(output_path, dpi=600)
        logging.info(f"Balanced accuracy plot saved to {output_path}")

    except Exception as e:
        logging.error(f"Error occurred while plotting balanced accuracies: {e}")


def parse_arguments() -> argparse.Namespace:
    """Parses command-line arguments.

    Returns:
        argparse.Namespace: Parsed command-line arguments.
    """
    argparser = argparse.ArgumentParser()
    argparser.add_argument(
        "--mode",
        type=str,
        default="BGC",
        help="Mode of operation, either 'BGC' or 'metabolism'",
    )
    argparser.add_argument(
        "--device",
        type=str,
        default="cuda",
        help="Device to use for computation ('cuda' or 'cpu')",
    )
    return argparser.parse_args()


def setup_device(device_name: str) -> torch.device:
    """Sets up the computation device.

    Args:
        device_name (str): Name of the device ('cuda' or 'cpu').

    Returns:
        torch.device: The device object to be used in training.
    """
    device = torch.device(device_name if torch.cuda.is_available() else "cpu")
    logging.info(f"Using device: {device}")
    return device


def initialize_classifiers(
    device: torch.device, num_columns: int, unique_count_target: int, num_fragments: int
) -> Tuple[List[str], List]:
    """Initializes classifiers including both PyTorch models and Scikit-Learn classifiers.

    Args:
        device (torch.device): The device to which PyTorch models will be moved.
        num_columns (int): Number of features in the dataset.
        unique_count_target (int): Number of unique classes in the target variable.

    Returns:
        Tuple[List[str], List]: A tuple containing a list of classifier names and a list of classifiers.
    """
    names_classifiers = [
        "ExtraTreesClassifier",
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

    models = [
        LSTM(
            in_features=num_columns - 1,
            hidden_size=20,
            num_fragments=num_fragments,
            num_classes=unique_count_target,
        ).to(device),
        BasicFFNN(num_classes=unique_count_target, in_features=num_columns - 1).to(
            device
        ),
        IntermediateFFNN(
            num_classes=unique_count_target, in_features=num_columns - 1
        ).to(device),
        AdvancedFFNN(num_classes=unique_count_target, in_features=num_columns - 1).to(
            device
        ),
        VeryAdvancedFFNN(
            num_classes=unique_count_target, in_features=num_columns - 1
        ).to(device),
    ]
    criteria = [nn.CrossEntropyLoss() for _ in models]
    optimizers = [optim.Adam(model.parameters(), lr=0.001) for model in models]

    classifiers = [
        ExtraTreesClassifier(max_depth=25, min_samples_leaf=1, class_weight="balanced"),
        (models[0], criteria[0], optimizers[0]),
        (models[1], criteria[1], optimizers[1]),
        (models[2], criteria[2], optimizers[2]),
        (models[3], criteria[3], optimizers[3]),
        (models[4], criteria[4], optimizers[4]),
        RandomForestClassifier(max_depth=25, n_estimators=10, max_features=1),
        AdaBoostClassifier(n_estimators=100),
        DecisionTreeClassifier(max_depth=25),
        MLPClassifier(
            solver="lbfgs", alpha=1e-5, hidden_layer_sizes=(5, 2), random_state=1
        ),
    ]

    return names_classifiers, classifiers


def prepare_output_directories(mode: str) -> Tuple[str, str, str]:
    """Prepares the output directories based on the mode.

    Args:
        mode (str): The mode of operation, either 'BGC' or 'metabolism'.

    Returns:
        Tuple[str, str, str]: A tuple containing the label mapping, directory of feature matrices, and output folder name.

    Raises:
        ValueError: If an invalid mode is provided.
    """
    if mode == "BGC":
        label_mapping = BGC_types
        directory_feature_matrices = (
            "../preprocessing/preprocessed_data/dataset_transformer_new"
        )
        foldername_output = "../classifiers/Transformer_BGC_type/"
    elif mode == "metabolism":
        label_mapping = ["primary_metabolism", "secondary_metabolism"]
        directory_feature_matrices = (
            "../preprocessing/preprocessed_data/dataset_transformer_new"
        )
        foldername_output = "../classifiers/Transformer_metabolism/"
    else:
        raise ValueError(f"MODE must be BGC or metabolism, not {mode}")

    logging.info(f"Mode: {mode}, Output directory: {foldername_output}")
    return label_mapping, directory_feature_matrices, foldername_output


def process_enzyme(
    enzyme: str,
    mode: str,
    directory_feature_matrices: str,
    device: torch.device,
    foldername_output: str,
    label_mapping: List[str],
    test_size: float,
) -> List[pd.DataFrame]:
    """Processes each enzyme by training classifiers and saving results.

    Args:
        enzyme (str): The name of the enzyme being processed.
        mode (str): The mode of operation, either 'BGC' or 'metabolism'.
        directory_feature_matrices (str): The directory containing the feature matrices.
        device (torch.device): The device to use for computation.
        foldername_output (str): The output directory for saving results.
        label_mapping (List[str]): The label mapping for the classes.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        List[pd.DataFrame]: A list of DataFrames containing metrics for each classifier.
    """
    if enzyme == "ycao" and mode == "BGC":
        logging.info(
            f"Skipping enzyme: {enzyme} as BGC classification is not applicable."
        )
        return []

    path_feature_matrix = os.path.join(
        directory_feature_matrices, f"{enzyme}_{mode}_type_feature_matrix.csv"
    )
    df = pd.read_csv(path_feature_matrix)
    df.replace([np.inf, -np.inf], np.nan, inplace=True)
    df = df.dropna()
    num_columns = df.shape[1]
    unique_count_target = df["target"].nunique()

    x_train, x_test, y_train, y_test, x_data, y_data = create_training_test_set(
        path_feature_matrix, test_size
    )
    num_fragments = len(enzymes[enzyme]["splitting_list"])
    # Re-initialize classifiers for the current enzyme
    names_classifiers, classifiers = initialize_classifiers(
        device, num_columns, unique_count_target, num_fragments
    )

    all_cross_validation_scores = {}
    all_balanced_accuracies = {}
    all_metrics = []

    for classifier, name_classifier in zip(classifiers, names_classifiers):
        logging.info(f"Training classifier: {name_classifier} for enzyme: {enzyme}")
        if isinstance(classifier, tuple):  # PyTorch models
            model, criterion, optimizer = classifier
            trainer = TrainerPytorch(
                model, criterion, optimizer, label_mapping, foldername_output
            )
            f1_macro, balanced_acc, auc_score, logloss, metrics = trainer.train(
                x_train, y_train, x_test, y_test, name_classifier, enzyme
            )
            all_metrics.append(metrics)
            all_balanced_accuracies[name_classifier + "_" + enzyme] = balanced_acc
        else:  # Scikit-Learn classifiers
            trainer = TrainerScikitLearn(label_mapping, foldername_output)
            cross_val_scores, balanced_accuracy = trainer.train(
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
            all_cross_validation_scores[name_classifier + "_" + enzyme] = (
                cross_val_scores
            )
            all_balanced_accuracies[name_classifier + "_" + enzyme] = balanced_accuracy

    plot_balanced_accuracies(foldername_output, all_balanced_accuracies, enzyme)
    return all_metrics


def main():
    """Main entry point for the script. Handles argument parsing, device setup, and the processing of enzymes."""
    args = parse_arguments()
    device = setup_device(args.device)
    label_mapping, directory_feature_matrices, foldername_output = (
        prepare_output_directories(args.mode)
    )

    all_metrics = []

    for enzyme in enzymes:
        metrics = process_enzyme(
            enzyme,
            args.mode,
            directory_feature_matrices,
            device,
            foldername_output,
            label_mapping,
            test_size=0.3,
        )
        if metrics:
            all_metrics.extend(metrics)

    if all_metrics:
        metrics_df = pd.concat(all_metrics, axis=0)
        date_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        metrics_df.to_csv(
            os.path.join(foldername_output, f"metrics_all_classifiers{date_time}.csv"),
            index=False,
        )
        logging.info(f"Metrics saved to {foldername_output}")


if __name__ == "__main__":
    main()
