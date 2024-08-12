import os
import sys
import subprocess
import pickle
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
from Bio import SeqIO, AlignIO
from Bio.Seq import Seq
from Bio.SeqRecord import SeqRecord
from sklearn.base import BaseEstimator
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    accuracy_score,
    balanced_accuracy_score,
)
import torch
from sklearn.preprocessing import LabelEncoder
import esm
import logging
from importlib.resources import files
import torch.nn as nn
import torch.nn.functional as F
from tailenza.classifiers.preprocessing.feature_generation import (
    fragment_alignment,
    featurize_fragments,
)
from tailenza.data.enzyme_information import enzymes, BGC_types

from tailenza.classifiers.machine_learning.machine_learning_training_classifiers_AA_BGC_type import (
    LSTM,
    BasicFFNN,
    IntermediateFFNN,
    AdvancedFFNN,
    VeryAdvancedFFNN,
)


def muscle_align_sequences(fasta_filename, enzyme, enzymes):
    """Align sequences using muscle and returns the alignment"""
    num_sequences = sum(1 for _ in SeqIO.parse(fasta_filename, "fasta"))
    if num_sequences <= 1:
        return AlignIO.read(open(fasta_filename), "fasta")

    muscle_cmd = [
        "muscle",
        "-in",
        fasta_filename,
        "-out",
        f"{fasta_filename}_aligned.fasta",
        "-seqtype",
        "protein",
        "-maxiters",
        "16",
        "-gapopen",
        str(enzymes[enzyme]["gap_opening_penalty"]),
        "-gapextend",
        str(enzymes[enzyme]["gap_extend_penalty"]),
        "-center",
        str(enzymes[enzyme]["center"]),
    ]

    try:
        subprocess.check_call(
            muscle_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        logging.error(f"Failed to run command {' '.join(muscle_cmd)}")
        sys.exit(1)

    return AlignIO.read(open(f"{fasta_filename}_aligned.fasta"), "fasta")


def classifier_prediction(feature_matrix, classifier_path, mode):
    """Predict values using a classifier"""
    label_encoder = LabelEncoder()
    if mode == "metabolism":
        label_encoder.classes_ = np.array(
            ["primary_metabolism", "secondary_metabolism"]
        )
        unique_count_target = 2
    elif mode == "BGC":
        label_encoder.classes_ = np.array(BGC_types)
        unique_count_target = len(BGC_types)
    else:
        raise ValueError(f"Mode {mode} not available")

    num_columns = feature_matrix.shape[1]
    logging.debug(f"Num columns: {num_columns}, classes: {unique_count_target}")
    classifier = None

    if os.path.splitext(classifier_path)[1] == ".pth":
        # Check which model type is in the classifier path and assign the appropriate class
        classifier = get_classifier_by_path(
            classifier_path, num_columns, unique_count_target
        )
        classifier.load_state_dict(torch.load(classifier_path))
        classifier.eval()
        classifier.to(device)
        with torch.no_grad():
            if isinstance(feature_matrix, pd.DataFrame):
                feature_matrix = feature_matrix.to_numpy()
            feature_matrix = torch.tensor(feature_matrix, dtype=torch.float32).to(
                device
            )
            logits = classifier(feature_matrix)

            predicted_values = torch.argmax(logits, dim=1).cpu().numpy()
            predicted_values = label_encoder.inverse_transform(predicted_values)
            score_predicted_values = F.softmax(logits, dim=1).cpu().numpy()
    else:
        with open(classifier_path, "rb") as file:
            classifier = pickle.load(file)

        predicted_values = classifier.predict(feature_matrix)
        score_predicted_values = classifier.predict_proba(feature_matrix)

    return predicted_values, score_predicted_values


def get_classifier_by_path(classifier_path, num_columns, unique_count_target):
    if "_LSTM" in classifier_path:
        return LSTM(
            in_features=num_columns, hidden_size=20, num_classes=unique_count_target
        )
    elif "_BasicFFNN" in classifier_path:
        return BasicFFNN(num_classes=unique_count_target, in_features=num_columns)
    elif "_IntermediateFFNN" in classifier_path:
        return IntermediateFFNN(
            num_classes=unique_count_target, in_features=num_columns
        )
    elif "_AdvancedFFNN" in classifier_path:
        return AdvancedFFNN(num_classes=unique_count_target, in_features=num_columns)
    elif "_VeryAdvancedFFNN" in classifier_path:
        return VeryAdvancedFFNN(
            num_classes=unique_count_target, in_features=num_columns
        )
    else:
        raise ValueError("Unknown model type in the path")


def plot_confusion_matrix(cm, labels, title="Confusion Matrix", save_path=None):
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def plot_classification_report(cr, title="Classification Report", save_path=None):
    lines = cr.split("\n")
    classes = []
    plot_data = []
    for line in lines[2 : len(lines) - 3]:
        line_data = line.split()
        classes.append(line_data[0])
        plot_data.append([float(x) for x in line_data[1 : len(line_data) - 1]])
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        plot_data,
        annot=True,
        fmt=".2f",
        cmap="Blues",
        xticklabels=["Precision", "Recall", "F1-score"],
        yticklabels=classes,
    )
    plt.title(title)
    plt.xlabel("Metrics")
    plt.ylabel("Classes")
    if save_path:
        plt.savefig(save_path)
    plt.show()


def save_classification_report(cr, save_path):
    with open(save_path, "w") as f:
        f.write(cr)


def get_true_BGC_label_from_filename(filename, BGC_types):
    """Extract true BGC label from the filename"""
    for BGC_type in BGC_types:
        if BGC_type in filename:
            return BGC_type
    return None


def setup_logging():
    logging.basicConfig(
        level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
    )
    logging.debug("Logging setup complete.")


def load_model_and_alphabet(file_path_model):
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(file_path_model)
    return model.eval().to(device), alphabet


def process_batch(
    batch, enzyme, BGC_type, model, batch_converter, device, include_charge_features
):
    reference_seq = enzymes[enzyme]["reference_for_alignment"]
    reference_record = SeqRecord(Seq(reference_seq), id="Reference")
    batch.append(reference_record)

    temp_fasta_filename = f"{enzyme}_{BGC_type}_batch_temp.fasta"
    with open(temp_fasta_filename, "w") as output_handle:
        SeqIO.write(batch, output_handle, "fasta")

    alignment = muscle_align_sequences(temp_fasta_filename, enzyme, enzymes)
    fragment_matrix = fragment_alignment(
        alignment, enzymes[enzyme]["splitting_list"], True
    )
    feature_matrix = featurize_fragments(
        fragment_matrix,
        batch_converter,
        model,
        include_charge_features,
        device,
    )
    os.remove(temp_fasta_filename)
    return feature_matrix


def main():
    setup_logging()
    global device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    package_dir = files("tailenza").joinpath("")
    logging.debug("Package directory: %s", package_dir)

    file_path_model = package_dir.joinpath("data", "esm1b_t33_650M_UR50S.pt")
    model, alphabet = load_model_and_alphabet(file_path_model)
    batch_converter = alphabet.get_batch_converter()

    classifier_dirs = {
        "BGC_type": "../classifiers/classifiers/Transformer_BGC_type/",
        "metabolism": "../classifiers/classifiers/Transformer_metabolism/",
    }

    include_charge_features = True

    # Aggregating all true and predicted labels across enzymes
    aggregated_true_BGC_labels = []
    aggregated_predicted_BGC_labels = []
    aggregated_true_metabolism_labels = []
    aggregated_predicted_metabolism_labels = []

    for enzyme in enzymes:
        for BGC_type in BGC_types:
            fasta_filename = f"validation_dataset/MIBiG_dataset/{enzyme}_MIBiG_{BGC_type}_without_too_long_too_short.fasta"
            if not os.path.isfile(fasta_filename):
                continue
            logging.info(fasta_filename)
            sequences = list(SeqIO.parse(fasta_filename, "fasta"))

            batch_size = 100
            batches = [
                sequences[i : i + batch_size]
                for i in range(0, len(sequences), batch_size)
            ]
            combined_feature_matrix = None

            for batch in batches:
                feature_matrix = process_batch(
                    batch,
                    enzyme,
                    BGC_type,
                    model,
                    batch_converter,
                    device,
                    include_charge_features,
                )
                if combined_feature_matrix is None:
                    combined_feature_matrix = feature_matrix
                else:
                    combined_feature_matrix = np.concatenate(
                        (combined_feature_matrix, feature_matrix), axis=0
                    )

            feature_matrix = combined_feature_matrix
            predicted_metabolisms, scores_predicted_metabolism = classifier_prediction(
                feature_matrix,
                os.path.join(
                    classifier_dirs["metabolism"],
                    f"{enzyme}{enzymes[enzyme]['classifier_metabolism']}",
                ),
                "metabolism",
            )

            if enzyme == "ycao":
                predicted_BGC_types = ["RiPP"] * len(combined_feature_matrix)
                scores_predicted_BGC_type = [1] * len(combined_feature_matrix)
            else:
                predicted_BGC_types, scores_predicted_BGC_type = classifier_prediction(
                    feature_matrix,
                    os.path.join(
                        classifier_dirs["BGC_type"],
                        f"{enzyme}{enzymes[enzyme]['classifier_BGC_type']}",
                    ),
                    "BGC",
                )

            true_BGC_label = [BGC_type] * len(combined_feature_matrix)
            aggregated_true_BGC_labels.extend(true_BGC_label)
            true_metabolism_label = ["secondary_metabolism"] * len(
                combined_feature_matrix
            )
            aggregated_true_metabolism_labels.extend(true_metabolism_label)
            aggregated_predicted_BGC_labels.extend(predicted_BGC_types)
            aggregated_predicted_metabolism_labels.extend(predicted_metabolisms)

    # Generate overall classification report and confusion matrix
    bgc_classification_report = classification_report(
        aggregated_true_BGC_labels, aggregated_predicted_BGC_labels, labels=BGC_types
    )
    logging.info(f"BGC Type Classification Report:\n{bgc_classification_report}")
    bgc_confusion_matrix = confusion_matrix(
        aggregated_true_BGC_labels, aggregated_predicted_BGC_labels, labels=BGC_types
    )
    logging.info(f"Confusion Matrix:\n{bgc_confusion_matrix}")

    metabolism_classification_report = classification_report(
        aggregated_true_metabolism_labels,
        aggregated_predicted_metabolism_labels,
        labels=["secondary_metabolism", "primary_metabolism"],
    )
    logging.info(
        f"Metabolism Classification Report:\n{metabolism_classification_report}"
    )
    metabolism_confusion_matrix = confusion_matrix(
        aggregated_true_metabolism_labels,
        aggregated_predicted_metabolism_labels,
        labels=["secondary_metabolism", "primary_metabolism"],
    )
    logging.info(f"Confusion Matrix:\n{metabolism_confusion_matrix}")

    # Plot and save confusion matrices
    plot_confusion_matrix(
        bgc_confusion_matrix,
        BGC_types,
        title="Overall BGC Type Confusion Matrix",
        save_path="confusion_matrices/overall_BGC_type_confusion_matrix.png",
    )
    save_classification_report(
        bgc_classification_report,
        "classification_reports/overall_BGC_type_classification_report.txt",
    )
    plot_confusion_matrix(
        metabolism_confusion_matrix,
        ["secondary_metabolism", "primary_metabolism"],
        title="Overall Metabolism Confusion Matrix",
        save_path="confusion_matrices/overall_metabolism_confusion_matrix.png",
    )
    save_classification_report(
        metabolism_classification_report,
        "classification_reports/overall_metabolism_classification_report.txt",
    )


if __name__ == "__main__":
    main()
