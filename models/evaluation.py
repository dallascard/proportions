import sys
import numpy as np

from sklearn.metrics import f1_score as skl_f1_score
from sklearn.metrics import accuracy_score as skl_acc_score


def evaluate(objective, true, pred, pred_probs, n_classes=2, pos_label=1, average=None, weights=None):
    if objective == 'f1':
        return f1_score(true, pred, n_classes=n_classes, pos_label=pos_label, average=average, weights=weights)
    elif objective == 'acc':
        return acc_score(true, pred, weights=weights)
    elif objective == 'calibration':
        return calibration_score_binary(true, pred_probs, weights=weights)
    else:
        sys.exit("objective not recognized")


def f1_score(true, pred, n_classes=2, pos_label=1, average=None, weights=None):
    """
    Override f1_score in sklearn in order to deal with both binary and multiclass cases
    :param true: true labels
    :param pred: predicted labels
    :param n_classes: total number of different possible labels
    :param pos_label: label to use as the positive label for the binary case (0 or 1)
    :param average: how to calculate f1 for the multiclass case (default = 'micro')

    :return: f1 score
    """

    if n_classes == 2:
        if np.sum(true * pred) == 0:
            f1 = 0.0
        else:
            f1 = skl_f1_score(true, pred, average='binary', labels=range(n_classes), pos_label=pos_label, sample_weight=weights)
    else:
        if average is None:
            f1 = skl_f1_score(true, pred, average='micro', labels=range(n_classes), pos_label=None, sample_weight=weights)
        else:
            f1 = skl_f1_score(true, pred, average=average, labels=range(n_classes), pos_label=None, sample_weight=weights)
    return f1


def acc_score(true, pred, weights=None):
    acc = skl_acc_score(np.array(true, dtype=int), np.array(pred, dtype=int), sample_weight=weights)
    return acc


def calibration_score_binary(binary_label_vector, pred_probs, weights, min_bins=3, max_bins=7):
    """
    assume binary data
    binary_label_vector: a vector of binary labels
    pred_probs: a matrix of predicted probabilities of the positive class [n x Y]
    weights: a vector of weights
    """

    n_items, n_classes = pred_probs.shape
    assert n_items == len(binary_label_vector)
    assert n_classes == 2

    # extract the probability of the positive class
    pred_probs = pred_probs[:, 1]

    order = np.argsort(pred_probs)

    averages = []
    for n_bins in range(min_bins, max_bins+1):
        if n_items < n_bins:
            n_bins = n_items
        breakpoints = list(np.array(np.arange(n_bins)/float(n_bins) * n_items, dtype=int).tolist()) + [n_items]

        mae = 0.0
        for b in range(n_bins):
            start = breakpoints[b]
            end = breakpoints[b+1]
            indices = order[start:end]
            mean_bin_probs = np.dot(pred_probs[indices], weights[indices]) / np.sum(weights[indices])
            mean_bin_labels = np.dot(binary_label_vector[indices], weights[indices]) / np.sum(weights[indices])
            ae = np.abs(mean_bin_labels - mean_bin_probs)
            mae += ae

        averages.append(mae / n_bins)

    return np.mean(averages)


def eval_proportions_mae_binary(binary_label_vector, pred_probs, weights):
    pred_prop = np.dot(pred_probs, weights) / np.sum(weights)
    true_prop = np.dot(binary_label_vector, weights) / np.sum(weights)
    ae = np.abs(true_prop - pred_prop)
    return ae


def compute_proportions_from_predicted_labels(predictions, instance_weights, n_classes=2):
    """
    Compute label proportion (weighted), using predicted labels, using given number of classes
    """
    n_items = len(predictions)
    if instance_weights is None:
        instance_weights = np.ones(n_items)
    class_counts = np.zeros(n_classes)
    for c in range(n_classes):
        items = np.array(predictions == c, dtype=bool)
        class_counts[c] = np.sum(instance_weights[items])
    proportions = class_counts / float(np.sum(class_counts))
    return proportions


