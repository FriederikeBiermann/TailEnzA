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
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
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
    ]

    try:
        subprocess.check_call(
            muscle_cmd, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL
        )
    except subprocess.CalledProcessError:
        print(f"Error: Failed to run command {' '.join(muscle_cmd)}")
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
    logging.debug(f"Num columns: {num_columns}, classes {unique_count_target}")
    model_class = None
    if os.path.splitext(classifier_path)[1] == ".pth":
        # Check which model type is in the classifier path and assign the appropriate class
        if "_LSTM" in classifier_path:
            classifier = LSTM(
                in_features=num_columns,
                hidden_size=20,
                num_classes=unique_count_target,
            )
        elif "_BasicFFNN" in classifier_path:
            classifier = BasicFFNN(
                num_classes=unique_count_target, in_features=num_columns
            )
        elif "_IntermediateFFNN" in classifier_path:
            classifier = IntermediateFFNN(
                num_classes=unique_count_target, in_features=num_columns
            )
        elif "_AdvancedFFNN" in classifier_path:
            classifier = AdvancedFFNN(
                num_classes=unique_count_target, in_features=num_columns
            )
        elif "_VeryAdvancedFFNN" in classifier_path:
            classifier = VeryAdvancedFFNN(
                num_classes=unique_count_target, in_features=num_columns
            )
        else:
            raise ValueError("Unknown model type in the path")

        classifier.load_state_dict(torch.load(classifier_path))
        classifier.eval()
        classifier.to(device)
        with torch.no_grad():
            if isinstance(feature_matrix, pd.DataFrame):
                feature_matrix = feature_matrix.to_numpy()
            feature_matrix = torch.tensor(
                feature_matrix, dtype=torch.float32
            ).to(device)
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


def plot_confusion_matrix(cm, labels, title="Confusion Matrix"):
    plt.figure(figsize=(10, 7))
    sns.heatmap(
        cm, annot=True, fmt="d", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.show()


def plot_classification_report(cr, title="Classification Report"):
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
    plt.show()


def get_true_BGC_label_from_filename(filename, BGC_types):
    """Extract true BGC label from the filename"""
    for BGC_type in BGC_types:
        if BGC_type in filename:
            return BGC_type
    return None


logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)
package_dir = files("tailenza").joinpath("")
logging.debug("Package directory: %s", package_dir)

directory_of_classifiers_BGC_type = "../classifiers/classifiers/BGC_type/"
directory_of_classifiers_NP_affiliation = "../classifiers/classifiers/metabolism_type/"


device = torch.device("cuda:1" if torch.cuda.is_available() else "cpu")
# Initialize lists for true and predicted labels
true_BGC_labels = []
predicted_BGC_labels = []
true_metabolism_labels = []
predicted_metabolism_labels = []

include_charge_features = True


# Process each enzyme
def main():
    file_path_model = package_dir.joinpath("data", "esm1b_t33_650M_UR50S.pt")
    model, alphabet = esm.pretrained.load_model_and_alphabet_local(file_path_model)
    model = model.eval().to(device)
    batch_converter = alphabet.get_batch_converter()
    for enzyme in enzymes:
        for BGC_type in BGC_types:
            fasta_filename = f"validation_dataset/MIBiG_dataset/{enzyme}_MIBiG_{BGC_type}_without_too_long_too_short.fasta"
            if not os.path.isfile(fasta_filename):
                print("not_found")
                continue
            sequences = list(SeqIO.parse(fasta_filename, "fasta"))
    
            # Add the reference sequence
            reference_seq = enzymes[enzyme]["reference_for_alignment"]
            reference_record = SeqRecord(Seq(reference_seq), id="Reference")
            batch_size = 100
            # Split the sequences into batches of specified size
            batches = [sequences[i:i + batch_size] for i in range(0, len(sequences), batch_size)]

            # Placeholder to store feature matrices
            combined_feature_matrix = None

            for i, batch in enumerate(batches):
                # Add the reference sequence to the current batch
                batch.append(reference_record)

                # Create a temporary FASTA file for the current batch
                temp_fasta_filename = f"{enzyme}_batch_{i+1}.fasta"
                with open(temp_fasta_filename, "w") as output_handle:
                    SeqIO.write(batch, output_handle, "fasta")
        
                # Align the sequences in the batch
                alignment = muscle_align_sequences(temp_fasta_filename, enzyme, enzymes)
                print(f"Alignment for batch {i+1}:", alignment, enzymes[enzyme]["splitting_list"])
        
                # Fragment the alignment
                fragment_matrix = fragment_alignment(
                    alignment, enzymes[enzyme]["splitting_list"], True
                )
                print(f"Fragment matrix for batch {i+1}:", fragment_matrix)
        
                # Featurize the fragments
                feature_matrix = featurize_fragments(
                    fragment_matrix, batch_converter, model, include_charge_features, device
                )

                # Concatenate the feature matrix for the batch with the combined feature matrix
                if combined_feature_matrix is None:
                    combined_feature_matrix = feature_matrix
                else:
                    combined_feature_matrix = np.concatenate((combined_feature_matrix, feature_matrix), axis=0)
        
                # Remove the temporary FASTA file
                os.remove(temp_fasta_filename)

            feature_matrix = combined_feature_matrix
            predicted_metabolisms = {}
            scores_predicted_metabolism = {}
            predicted_BGC_types = {}
            scores_predicted_BGC_type = {}
            metabolism_classifier_path = os.path.join(
                directory_of_classifiers_NP_affiliation,
                f"{enzyme}{enzymes[enzyme]['classifier_metabolism']}",
            )
            BGC_type_classifier_path = os.path.join(
                directory_of_classifiers_BGC_type,
                f"{enzyme}{enzymes[enzyme]['classifier_BGC_type']}",
            )

            (
                predicted_metabolisms,
                scores_predicted_metabolism,
            ) = classifier_prediction(
                feature_matrix, metabolism_classifier_path, "metabolism"
            )
            if enzyme == "ycao":
                predicted_BGC_types = ["RiPP"] * len(combined_feature_matrix)
                scores_predicted_BGC_type = [1] * len(combined_feature_matrix)
            else:
                (
                    predicted_BGC_types,
                    scores_predicted_BGC_type,
                ) = classifier_prediction(feature_matrix, BGC_type_classifier_path, "BGC")

            true_BGC_label = [BGC_type] * len(combined_feature_matrix)
            true_BGC_labels.extend(true_BGC_label)
            true_metabolism_label = ["secondary metabolism"] * len(combined_feature_matrix)
            true_metabolism_labels.extend(true_metabolism_label)
            predicted_BGC_labels.extend(predicted_BGC_types)
            predicted_metabolism_labels.extend(predicted_metabolisms)

    # Calculate and print metrics for BGC type classification
    print("BGC Type Classification Report:")
    bgc_classification_report = classification_report(
        true_BGC_labels, predicted_BGC_labels, labels=BGC_types
    )
    print(bgc_classification_report)
    bgc_confusion_matrix = confusion_matrix(
        true_BGC_labels, predicted_BGC_labels, labels=BGC_types
    )
    print("Confusion Matrix:\n", bgc_confusion_matrix)

    # Calculate and print metrics for metabolism classification
    print("Metabolism Classification Report:")
    metabolism_classification_report = classification_report(
        true_metabolism_labels,
        predicted_metabolism_labels,
        labels=["secondary metabolism"],
    )
    print(metabolism_classification_report)
    metabolism_confusion_matrix = confusion_matrix(
        true_metabolism_labels,
        predicted_metabolism_labels,
        labels=["secondary metabolism"],
    )
    print("Confusion Matrix:\n", metabolism_confusion_matrix)

    # Plot confusion matrices
    plot_confusion_matrix(
        bgc_confusion_matrix, BGC_types, title="BGC Type Confusion Matrix"
    )
    plot_confusion_matrix(
        metabolism_confusion_matrix,
        ["secondary metabolism"],
        title="Metabolism Confusion Matrix",
    )

    # Plot classification reports
    plot_classification_report(
        bgc_classification_report, title="BGC Type Classification Report"
    )
    plot_classification_report(
        metabolism_classification_report, title="Metabolism Classification Report"
    )
if __name__ == "__main__":
    main()
