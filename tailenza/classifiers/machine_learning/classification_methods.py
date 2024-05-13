import pandas as pd
from sklearn.metrics import classification_report, balanced_accuracy_score
from sklearn.model_selection import cross_val_score, train_test_split
from imblearn.over_sampling import RandomOverSampler
import pickle
import matplotlib.pyplot as plt
from sklearn.metrics import (
    balanced_accuracy_score,
    classification_report,
    plot_confusion_matrix,
    cross_val_score,
    f1_score,
    log_loss,
    roc_auc_score,
    confusion_matrix,
)
from sklearn.preprocessing import LabelBinarizer
import seaborn as sns
import numpy as np
import os
import logging
import torch
from torch import TensorDataset, DataLoader


ros = RandomOverSampler(random_state=0)


def train_pytorch_classifier(
    model, criterion, optimizer, x_train, y_train, x_test, y_test, epochs=50
):
    model.train()
    dataset = TensorDataset(
        torch.tensor(x_train.astype(np.float32)), torch.tensor(y_train.astype(np.int64))
    )
    loader = DataLoader(dataset, batch_size=10, shuffle=True)

    for epoch in range(epochs):
        for inputs, targets in loader:
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, targets)
            loss.backward()
            optimizer.step()

    # Evaluate the model
    model.eval()
    with torch.no_grad():
        y_pred = model(torch.tensor(x_test.astype(np.float32))).argmax(dim=1).numpy()
        f1_macro = f1_score(y_test, y_pred, average="macro")
        f1_micro = f1_score(y_test, y_pred, average="micro")
        f1_weighted = f1_score(y_test, y_pred, average="weighted")
        balanced_acc = balanced_accuracy_score(y_test, y_pred)

        # Generate probability outputs for ROC and log loss
        outputs = model(torch.tensor(x_test.astype(np.float32)))
        softmax = nn.Softmax(dim=1)
        prob_outputs = softmax(outputs).numpy()

        # Handling different class scenarios for AUC
        lb = LabelBinarizer()
        y_test_binarized = lb.fit_transform(y_test)
        if y_test_binarized.shape[1] > 1:
            auc_score = roc_auc_score(
                y_test_binarized, prob_outputs, multi_class="ovr", average="macro"
            )
        else:
            auc_score = roc_auc_score(y_test, prob_outputs[:, 1])

        logloss = log_loss(y_test, prob_outputs)

        # Log results
        logging.info(f"Balanced Accuracy Score: {balanced_acc}")
        logging.info(f"F1 Score Macro: {f1_macro}")
        logging.info(f"F1 Score Micro: {f1_micro}")
        logging.info(f"F1 Score Weighted: {f1_weighted}")
        logging.info(f"AUC Score: {auc_score}")
        logging.info(f"Log Loss: {logloss}")

    return f1_macro, balanced_acc, auc_score, logloss


def plot_balanced_accuracies(foldernameoutput, all_balanced_accuracies, enzyme):
    labels = list(all_balanced_accuracies.keys())
    scores = list(all_balanced_accuracies.values())

    plt.figure(figsize=(14, 10))
    plt.bar(labels, scores, align="center", alpha=0.7)
    plt.ylabel("Balanced Accuracy Score")
    plt.title("Balanced Accuracy of Different Classifiers")
    plt.xticks(rotation=45)
    plt.tight_layout()

    # Save the figure
    plt.savefig(f"{foldernameoutput}{enzyme}balanced_accuracies.png", dpi=600)


def plot_feature_importance(importance, names, model_type, enzyme, foldernameoutput):
    # Create arrays from feature importance and feature names
    feature_importance = np.array(importance)
    feature_names = np.array(names)

    # Create a DataFrame using a Dictionary
    data = {"feature_names": feature_names, "feature_importance": feature_importance}
    fi_df = pd.DataFrame(data)

    # Sort the DataFrame in order decreasing feature importance
    fi_df.sort_values(by=["feature_importance"], ascending=False, inplace=True)

    # Plot feature importance
    plt.figure(figsize=(10, 8))
    sns.barplot(x=fi_df["feature_importance"], y=fi_df["feature_names"])
    plt.title(model_type + " - Feature Importance for enzyme " + enzyme)
    # Save the figure
    plt.xlabel("Feature Importance")
    plt.ylabel("Feature Names")
    plt.savefig(
        f"{foldernameoutput}{enzyme}_{model_type}_feature_importance.png", dpi=600
    )


def plot_cross_val_scores_with_variance(
    all_cross_validation_scores, foldernameoutput, enzyme
):
    labels = list(all_cross_validation_scores.keys())
    means = [np.mean(scores) for scores in all_cross_validation_scores.values()]
    std_devs = [np.std(scores) for scores in all_cross_validation_scores.values()]

    plt.figure(figsize=(14, 10))
    plt.bar(labels, means, yerr=std_devs, align="center", alpha=0.7, capsize=10)
    plt.ylabel("F1 Macro Score")
    plt.title("Cross Validation Scores of Different Classifiers")
    plt.xticks(rotation=45)
    plt.tight_layout()
    plt.savefig(f"{foldernameoutput}{enzyme}cross_val_scores.png", dpi=600)
    plt.show()


def plot_confusion_matrix(
    y_true, y_pred, enzyme, classifier_name, labels, foldernameoutput
):
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    plt.figure(figsize=(8, 6))
    sns.heatmap(
        cm, annot=True, fmt="g", cmap="Blues", xticklabels=labels, yticklabels=labels
    )
    plt.xlabel("Predicted labels")
    plt.ylabel("True labels")
    plt.title("Confusion Matrix for " + classifier_name + " on enzyme " + enzyme)
    plt.savefig(
        f"{foldernameoutput}{enzyme}_{classifier_name}_confusion_matrix.png", dpi=600
    )


def create_training_test_set(path_feature_matrix, test_size):
    # create training and test set from feature matrix
    feature_matrix = pd.read_csv(path_feature_matrix)
    feature_matrix = feature_matrix.sample(frac=1)
    # define target and features
    x_data = feature_matrix.loc[:, feature_matrix.columns != "target"]
    y_data = feature_matrix["target"]
    # split into training and test set
    x_train, x_test, y_train, y_test = train_test_split(
        x_data, y_data, test_size=test_size, shuffle=True
    )
    # resample to balance
    x_train, y_train = ros.fit_resample(x_train, y_train)
    return x_train, x_test, y_train, y_test, x_data, y_data


def train_classifier_and_get_accuracies(
    classifier,
    name_classifier,
    enzyme,
    x_data,
    y_data,
    x_train,
    y_train,
    x_test,
    y_test,
    foldernameoutput,
    BGC_types,
):
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
    plot_confusion_matrix(
        y_test,
        test_predict_classifier,
        enzyme,
        name_classifier,
        BGC_types,
        foldernameoutput,
    )
    plot_cross_val_scores_with_variance(
        cross_validation_classifier, foldernameoutput, enzyme
    )
    # if hasattr(classifier, 'feature_importances_'):
    # plot_feature_importance(classifier.feature_importances_, x_data.columns, name_classifier, enzyme, foldernameoutput)

    return cross_validation_classifier, balanced_accuracy_classifier


def optimize_leaf_number(
    classifier,
    name_classifier,
    enzyme,
    x_data,
    x_train,
    y_train,
    x_test,
    y_test,
    foldernameoutput,
):

    balanced_accuracy = 0.50
    # determine best mnimum number of leafes
    leafdiagr = pd.DataFrame(columns=["Minimum samples per leaf", "Balanced accuracy"])
    for minleaf in range(1, 5):

        classifier = classifier.fit(x_train, y_train)
        test_predict_classifier = classifier.predict(x_test)
        balanced_accuracy_new = balanced_accuracy_score(y_test, test_predict_classifier)
        new_line = {
            "Minimum samples per leaf": minleaf,
            "Balanced accuracy": balanced_accuracy_new,
        }
        leafdiagr = leafdiagr.append(new_line, ignore_index=True)
        if balanced_accuracy_new > balanced_accuracy:
            bestminleaf = minleaf
            balanced_accuracy = balanced_accuracy_new
        logging.info(leafdiagr)
    logging.info("Best minimum samples per leaf:", bestminleaf)

    # plot diagram of best minleaf
    plt.plot(
        "Minimum samples per leaf", "Balanced accuracy", data=leafdiagr, color="black"
    )
    plt.xlabel("Minimum samples per leaf")
    plt.ylabel("Balanced accuracy")
    plt.savefig(
        foldernameoutput + "_" + enzyme + "_" + name_classifier + "leafdiagr.png",
        format="png",
    )
    plt.show()
    return bestminleaf


def optimize_depth_classifier(
    classifier,
    name_classifier,
    enzyme,
    foldernameoutput,
    x_train,
    y_train,
    x_test,
    y_test,
):
    balanced_accuracy = 0.50
    # determine best mnimum number of leafes
    depthdiagr = pd.DataFrame(
        columns=["Maximal depth of random forest", "Balanced accuracy"]
    )
    for maximum_depth in range(10, 20):

        classifier = classifier.fit(x_train, y_train)
        test_predict_classifier = classifier.predict(x_test)
        balanced_accuracy_new = balanced_accuracy_score(y_test, test_predict_classifier)
        new_line = {
            "Maximal depth": maximum_depth,
            "Balanced accuracy": balanced_accuracy,
        }
        depthdiagr = depthdiagr.append(new_line, ignore_index=True)
        if balanced_accuracy_new > balanced_accuracy:
            bestmaximum_depth = maximum_depth
            balanced_accuracy = balanced_accuracy_new

    print("Best Max depth:", bestmaximum_depth)

    # plot diagram of best minleaf
    plt.plot("Maximal depth", "Balanced accuracy", data=depthdiagr, color="black")
    plt.xlabel("Maximal depth")
    plt.ylabel("Balanced accuracy")
    plt.savefig(
        foldernameoutput + "_" + enzyme + "_" + name_classifier + "depthdiagr.png",
        format="png",
    )
    plt.show()
    return bestmaximum_depth
