import os
import logging
import torch
import pickle
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime
from typing import List, Tuple, Dict, Union, Any
from torch import nn
from torch.optim import Optimizer
from torch.optim.lr_scheduler import ReduceLROnPlateau
from torch.utils.data import TensorDataset, DataLoader
from torch.utils.tensorboard import SummaryWriter
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
    ConfusionMatrixDisplay,
)

from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.preprocessing import LabelBinarizer, LabelEncoder
from imblearn.over_sampling import RandomOverSampler

# Check if GPU is available
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

ros = RandomOverSampler(random_state=0)


def create_training_test_set(path_feature_matrix: str, test_size: float) -> Tuple[
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
    Union[np.ndarray, None],
]:
    """
    Creates training and test sets from the feature matrix.

    Args:
        path_feature_matrix (str): The file path to the feature matrix CSV.
        test_size (float): The proportion of the dataset to include in the test split.

    Returns:
        Tuple[Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None], Union[np.ndarray, None]]:
        Returns six numpy arrays: x_train, x_test, y_train, y_test, x_data, and y_data.
        If an error occurs, all returned values will be None.
    """
    try:
        logging.info(f"Reading feature matrix from {path_feature_matrix}")

        # Read the feature matrix
        feature_matrix = pd.read_csv(path_feature_matrix)
        initial_row_count = feature_matrix.shape[0]
        logging.debug(
            f"Initial number of rows in the feature matrix: {initial_row_count}"
        )

        # Shuffle the feature matrix
        feature_matrix = feature_matrix.sample(frac=1).reset_index(drop=True)
        logging.info(f"Feature matrix shuffled")

        # Define target and features
        x_data = feature_matrix.loc[:, feature_matrix.columns != "target"].to_numpy()
        y_data = feature_matrix["target"].to_numpy()
        logging.debug(f"Feature matrix split into features and target arrays")

        # Split into training and test sets
        x_train, x_test, y_train, y_test = train_test_split(
            x_data, y_data, test_size=test_size, shuffle=True, stratify=y_data
        )
        logging.info(
            f"Data split into training and test sets with test size = {test_size}"
        )

        # Resample to balance the training data
        x_train_resampled, y_train_resampled = ros.fit_resample(x_train, y_train)
        logging.info(f"Training data resampled to balance classes")

        return x_train_resampled, x_test, y_train_resampled, y_test, x_data, y_data

    except Exception as e:
        logging.error(f"Error occurred while creating training and test sets: {e}")
        return None, None, None, None, None, None


class Trainer:
    """A base class for training classifiers.

    Attributes:
        label_mapping (List[str]): A list of class labels.
        output_dir (str): Directory where outputs will be saved.
        ros (RandomOverSampler): An instance of RandomOverSampler for handling imbalanced datasets.
        label_encoder (LabelEncoder): An instance of LabelEncoder to encode labels.
    """

    def __init__(self, label_mapping: List[str], output_dir: str = "output") -> None:
        """Initializes the Trainer with given parameters.

        Args:
            label_mapping (List[str]): A list of class labels.
            output_dir (str, optional): Directory where outputs will be saved. Defaults to 'output'.
        """
        self.label_mapping = label_mapping
        self.output_dir = output_dir
        self.ros = RandomOverSampler(random_state=0)
        self.label_encoder = LabelEncoder()
        self.label_encoder.classes_ = np.array(self.label_mapping)

    def _prepare_data(
        self,
        x_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        x_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Prepares the training and testing data by resampling and encoding.

        Args:
            x_train (Union[np.ndarray, pd.DataFrame]): Training features.
            y_train (Union[np.ndarray, pd.Series]): Training labels.
            x_test (Union[np.ndarray, pd.DataFrame]): Testing features.
            y_test (Union[np.ndarray, pd.Series]): Testing labels.

        Returns:
            Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]: Resampled and encoded training features and labels, and testing features and labels.
        """
        if not isinstance(x_train, np.ndarray):
            x_train = x_train.to_numpy()
            y_train = y_train.to_numpy()
            x_test = x_test.to_numpy()
            y_test = y_test.to_numpy()

        # Apply RandomOverSampler
        x_train_resampled, y_train_resampled = self.ros.fit_resample(x_train, y_train)
        y_train_resampled_encoded = self.label_encoder.transform(y_train_resampled)
        y_test_encoded = self.label_encoder.transform(y_test)

        return x_train_resampled, y_train_resampled_encoded, x_test, y_test_encoded

    def _log_initial_setup(self, name_classifier: str, enzyme: str) -> None:
        """Sets up initial logging and output directories.

        Args:
            name_classifier (str): The name of the classifier.
            enzyme (str): The enzyme or dataset identifier.
        """
        self.output_dir = os.path.join(self.output_dir, f"{enzyme}_{name_classifier}")
        os.makedirs(self.output_dir, exist_ok=True)

    def _plot_confusion_matrix(
        self,
        y_true: np.ndarray,
        y_pred: np.ndarray,
        enzyme: str,
        classifier_name: str,
        labels: List[str],
    ) -> None:
        """Plots and saves the confusion matrix.

        Args:
            y_true (np.ndarray): True labels.
            y_pred (np.ndarray): Predicted labels.
            enzyme (str): The enzyme or dataset identifier.
            classifier_name (str): The name of the classifier.
            labels (List[str]): List of class labels.
        """
        cm = confusion_matrix(y_true, y_pred, labels=labels)
        plt.figure(figsize=(8, 6))
        sns.heatmap(
            cm,
            annot=True,
            fmt="g",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
        )
        plt.xlabel("Predicted labels")
        plt.ylabel("True labels")
        plt.title("Confusion Matrix for " + classifier_name + " on enzyme " + enzyme)
        plt.savefig(
            f"{self.output_dir}/{enzyme}_{classifier_name}_confusion_matrix.png",
            dpi=600,
        )
        plt.close()


class TrainerPytorch(Trainer):
    """A class for training PyTorch models, inheriting from Trainer."""

    def __init__(
        self,
        model: nn.Module,
        criterion: nn.Module,
        optimizer: Optimizer,
        label_mapping: List[str],
        output_dir: str = "output",
        patience: int = 10,
        batch_size: int = 10,
        epochs: int = 100,
    ) -> None:
        """Initializes the TrainerPytorch with given parameters.

        Args:
            model (nn.Module): The PyTorch model to be trained.
            criterion (nn.Module): The loss function.
            optimizer (Optimizer): The optimizer for training the model.
            label_mapping (List[str]): A list of class labels.
            output_dir (str, optional): Directory where outputs will be saved. Defaults to 'output'.
            patience (int, optional): Number of epochs with no improvement after which training will be stopped. Defaults to 10.
            batch_size (int, optional): Number of samples per batch. Defaults to 10.
            epochs (int, optional): Number of training epochs. Defaults to 100.
        """
        super().__init__(label_mapping, output_dir)
        self.model = model.to(device)
        self.criterion = criterion
        self.optimizer = optimizer
        self.patience = patience
        self.batch_size = batch_size
        self.epochs = epochs
        self.writer: Union[SummaryWriter, None] = None

    def _log_initial_setup(self, name_classifier: str, enzyme: str) -> None:
        """Sets up initial logging, TensorBoard writer, and output directories.

        Args:
            name_classifier (str): The name of the classifier.
            enzyme (str): The enzyme or dataset identifier.
        """
        super()._log_initial_setup(name_classifier, enzyme)
        date_time = datetime.now().strftime("%Y-%m-%d-%H:%M:%S")
        log_dir = f"runs/{date_time}_lecun_{name_classifier}_{enzyme}"
        self.writer = SummaryWriter(log_dir=log_dir)
        logging.info(f"Processing model {name_classifier}")

    def _save_model(self, name_classifier: str, enzyme: str) -> None:
        """Saves the trained model.

        Args:
            name_classifier (str): The name of the classifier.
            enzyme (str): The enzyme or dataset identifier.
        """
        model_path = os.path.join(self.output_dir, f"{name_classifier}_model.pth")
        torch.save(self.model.state_dict(), model_path)

    def _plot_and_save(
        self, loss_values: List[float], name_classifier: str, enzyme: str
    ) -> None:
        """Plots and saves the training loss curve.

        Args:
            loss_values (List[float]): List of loss values for each batch.
            name_classifier (str): The name of the classifier.
            enzyme (str): The enzyme or dataset identifier.
        """
        plt.figure(figsize=(10, 5))
        plt.plot(loss_values, label="Training Loss")
        plt.xlabel("Batch")
        plt.ylabel("Loss")
        plt.title("Training Loss Curve")
        plt.legend()
        loss_curve_path = os.path.join(
            self.output_dir, f"loss_curve_{name_classifier}_{enzyme}.png"
        )
        plt.savefig(loss_curve_path)
        plt.close()

    def train(
        self,
        x_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        x_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        name_classifier: str,
        enzyme: str,
    ) -> Tuple[float, float, float, float, pd.DataFrame]:
        """Trains the PyTorch model and evaluates its performance.

        Args:
            x_train (Union[np.ndarray, pd.DataFrame]): Training features.
            y_train (Union[np.ndarray, pd.Series]): Training labels.
            x_test (Union[np.ndarray, pd.DataFrame]): Testing features.
            y_test (Union[np.ndarray, pd.Series]): Testing labels.
            name_classifier (str): The name of the classifier.
            enzyme (str): The enzyme or dataset identifier.

        Returns:
            Tuple[float, float, float, float, pd.DataFrame]: Returns evaluation metrics including F1 macro, balanced accuracy, AUC score, log loss, and a DataFrame with metrics.
        """
        self._log_initial_setup(name_classifier, enzyme)

        x_train_resampled, y_train_resampled_encoded, x_test, y_test_encoded = (
            self._prepare_data(x_train, y_train, x_test, y_test)
        )
        dataset = TensorDataset(
            torch.tensor(x_train_resampled.astype(np.float32)).to(device),
            torch.tensor(y_train_resampled_encoded.astype(np.int64)).to(device),
        )
        loader = DataLoader(dataset, batch_size=self.batch_size, shuffle=True)
        scheduler = ReduceLROnPlateau(self.optimizer, "min", patience=5)

        best_loss = float("inf")
        trigger_times = 0
        loss_values = []

        for epoch in range(self.epochs):
            self.model.train()
            running_loss = 0.0
            for i, (inputs, targets) in enumerate(loader):
                inputs, targets = inputs.to(device), targets.to(device)
                if i == 0 and epoch == 0:
                    self.writer.add_graph(self.model, inputs)
                self.optimizer.zero_grad()
                outputs = self.model(inputs)
                loss = self.criterion(outputs, targets)
                loss.backward()
                self.optimizer.step()

                running_loss += loss.item()
                if i % 10 == 9:  # Log every 10 batches
                    self.writer.add_scalar(
                        "Training Loss", running_loss / 10, epoch * len(loader) + i
                    )
                    loss_values.append(running_loss / 10)
                    running_loss = 0.0

            for name, param in self.model.named_parameters():
                self.writer.add_histogram(name, param, epoch)

            # Early stopping
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(
                    torch.tensor(x_test.astype(np.float32)).to(device)
                )
                val_loss = self.criterion(
                    val_outputs,
                    torch.tensor(y_test_encoded.astype(np.int64)).to(device),
                ).item()
                scheduler.step(val_loss)

                if val_loss < best_loss:
                    best_loss = val_loss
                    trigger_times = 0
                else:
                    trigger_times += 1
                    if trigger_times >= self.patience:
                        logging.info(f"Early stopping at epoch {epoch}")
                        break

        # Evaluation
        self._plot_and_save(loss_values, name_classifier, enzyme)
        self._save_model(name_classifier, enzyme)
        self.writer.close()
        return self.evaluate(x_test, y_test_encoded, name_classifier, enzyme, epoch)

    def evaluate(
        self,
        x_test: np.ndarray,
        y_test_encoded: np.ndarray,
        name_classifier: str,
        enzyme: str,
        epoch: int,
    ) -> Tuple[float, float, float, float, pd.DataFrame]:
        """Evaluates the trained PyTorch model.

        Args:
            x_test (np.ndarray): Testing features.
            y_test_encoded (np.ndarray): Encoded testing labels.
            name_classifier (str): The name of the classifier.
            enzyme (str): The enzyme or dataset identifier.
            epoch (int): The current epoch number.

        Returns:
            Tuple[float, float, float, float, pd.DataFrame]: Returns evaluation metrics including F1 macro, balanced accuracy, AUC score, log loss, and a DataFrame with metrics.
        """
        self.model.eval()
        with torch.no_grad():
            y_pred = (
                self.model(torch.tensor(x_test.astype(np.float32)).to(device))
                .argmax(dim=1)
                .cpu()
                .numpy()
            )
            f1_macro = f1_score(y_test_encoded, y_pred, average="macro")
            f1_micro = f1_score(y_test_encoded, y_pred, average="micro")
            f1_weighted = f1_score(y_test_encoded, y_pred, average="weighted")
            balanced_acc = balanced_accuracy_score(y_test_encoded, y_pred)

            outputs = self.model(torch.tensor(x_test.astype(np.float32)).to(device))
            softmax = nn.Softmax(dim=1)
            prob_outputs = softmax(outputs).cpu().numpy()

            lb = LabelBinarizer()
            y_test_binarized = lb.fit_transform(y_test_encoded)
            if y_test_binarized.shape[1] > 1:
                auc_score = roc_auc_score(
                    y_test_binarized, prob_outputs, multi_class="ovr", average="macro"
                )
            else:
                auc_score = roc_auc_score(y_test_encoded, prob_outputs[:, 1])

            logloss = log_loss(y_test_encoded, prob_outputs)

            self.writer.add_scalar("Balanced Accuracy Score", balanced_acc, epoch)
            self.writer.add_scalar("F1 Score Macro", f1_macro, epoch)
            self.writer.add_scalar("F1 Score Micro", f1_micro, epoch)
            self.writer.add_scalar("F1 Score Weighted", f1_weighted, epoch)
            self.writer.add_scalar("AUC Score", auc_score, epoch)
            self.writer.add_scalar("Log Loss", logloss, epoch)

            logging.info(f"Name model: {name_classifier}")
            logging.info(f"Balanced Accuracy Score: {balanced_acc}")
            logging.info(f"F1 Score Macro: {f1_macro}")
            logging.info(f"F1 Score Micro: {f1_micro}")
            logging.info(f"F1 Score Weighted: {f1_weighted}")
            logging.info(f"AUC Score: {auc_score}")
            logging.info(f"Log Loss: {logloss}")

            y_pred_decoded = self.label_encoder.inverse_transform(y_pred)
            y_test = self.label_encoder.inverse_transform(y_test_encoded)
            self._plot_confusion_matrix(
                y_test,
                y_pred_decoded,
                enzyme,
                name_classifier,
                self.label_mapping,
            )

            metrics_df = pd.DataFrame(
                [
                    {
                        "Enzyme": enzyme,
                        "Classifier": name_classifier,
                        "Epoch": epoch,
                        "Balanced Accuracy": balanced_acc,
                        "F1 Macro": f1_macro,
                        "F1 Micro": f1_micro,
                        "F1 Weighted": f1_weighted,
                        "AUC Score": auc_score,
                        "Log Loss": logloss,
                    }
                ]
            )

        return f1_macro, balanced_acc, auc_score, logloss, metrics_df


class TrainerScikitLearn(Trainer):
    """A class for training Scikit-Learn models, inheriting from Trainer."""

    def train(
        self,
        classifier: Any,
        name_classifier: str,
        enzyme: str,
        x_data: Union[np.ndarray, pd.DataFrame],
        y_data: Union[np.ndarray, pd.Series],
        x_train: Union[np.ndarray, pd.DataFrame],
        y_train: Union[np.ndarray, pd.Series],
        x_test: Union[np.ndarray, pd.DataFrame],
        y_test: Union[np.ndarray, pd.Series],
        foldernameoutput: str,
        BGC_types: List[str],
    ) -> Tuple[np.ndarray, float]:
        """Trains the Scikit-Learn model and evaluates its performance.

        Args:
            classifier (Any): The Scikit-Learn classifier to be trained.
            name_classifier (str): The name of the classifier.
            enzyme (str): The enzyme or dataset identifier.
            x_data (Union[np.ndarray, pd.DataFrame]): The full dataset features.
            y_data (Union[np.ndarray, pd.Series]): The full dataset labels.
            x_train (Union[np.ndarray, pd.DataFrame]): Training features.
            y_train (Union[np.ndarray, pd.Series]): Training labels.
            x_test (Union[np.ndarray, pd.DataFrame]): Testing features.
            y_test (Union[np.ndarray, pd.Series]): Testing labels.
            foldernameoutput (str): The folder name where outputs will be saved.
            BGC_types (List[str]): The list of BGC (Biosynthetic Gene Cluster) types.

        Returns:
            Tuple[np.ndarray, float]: Returns cross-validation scores and balanced accuracy.
        """
        classifier = classifier.fit(x_train, y_train)
        # Predict for test set
        test_predict_classifier = classifier.predict(x_test)

        # Calculate accuracy
        cross_validation_classifier = cross_val_score(
            classifier, x_data, y_data, cv=5, scoring="f1_macro"
        )
        balanced_accuracy_classifier = balanced_accuracy_score(
            y_test, test_predict_classifier
        )
        f1_macro = f1_score(y_test, test_predict_classifier, average="macro")
        f1_micro = f1_score(y_test, test_predict_classifier, average="micro")
        f1_weighted = f1_score(y_test, test_predict_classifier, average="weighted")

        # Log Loss and AUC Score
        if hasattr(classifier, "predict_proba"):
            y_probs = classifier.predict_proba(x_test)
            logloss = log_loss(y_test, y_probs)
            lb = LabelBinarizer()
            y_test_binarized = lb.fit_transform(y_test)
            if y_test_binarized.shape[1] > 1:
                auc_score = roc_auc_score(
                    y_test_binarized, y_probs, multi_class="ovr", average="macro"
                )
            else:
                auc_score = roc_auc_score(y_test, y_probs[:, 1])
            logging.info(f"{enzyme} {name_classifier} AUC Score: {auc_score}")
            logging.info(f"{enzyme} {name_classifier} Log Loss: {logloss}")

        # Logging results
        logging.info(
            f"{enzyme} {name_classifier} Score: {classifier.score(x_test, y_test)}"
        )
        logging.info(
            f"{enzyme} {name_classifier} Balanced Accuracy Score: {balanced_accuracy_classifier}"
        )
        logging.info(f"{enzyme} {name_classifier} F1 Score Macro: {f1_macro}")
        logging.info(f"{enzyme} {name_classifier} F1 Score Micro: {f1_micro}")
        logging.info(f"{enzyme} {name_classifier} F1 Score Weighted: {f1_weighted}")
        logging.info(
            f"{enzyme} {name_classifier} Crossvalidation Scores: {cross_validation_classifier}"
        )
        detailed_report = classification_report(
            y_test, test_predict_classifier, target_names=BGC_types
        )
        logging.info(
            f"{enzyme} {name_classifier} Detailed Classification Report:\n{detailed_report}"
        )

        # save trained classifier
        filename_classifier = os.path.join(
            foldernameoutput, f"{enzyme}_{name_classifier}_classifier.sav"
        )
        pickle.dump(classifier, open(filename_classifier, "wb"))
        # Plot everything
        self._plot_confusion_matrix(
            y_test,
            test_predict_classifier,
            enzyme,
            name_classifier,
            BGC_types,
        )
        return cross_validation_classifier, balanced_accuracy_classifier
