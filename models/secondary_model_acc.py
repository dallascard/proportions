import sys

import numpy as np

from models.evaluation import compute_proportions_from_predicted_labels


class ACC:
    """
    Secondary correction model to correct for label shift (ACC)
    """

    def __init__(self):
        self._p_pred_given_true = None
        self._model = None

    def fit(self, model, X, label_vector, weights):
        """
        compute a confusion matrix of p(y_hat=i|y=j) values
        For binary labels, the matrix contains the true positive rate and true negative rate
        :return: a matrix such that M[i,j] = p(y_hat = i | y = j)
        """
        self._model = model
        n_classes = model.get_n_classes()
        self._p_pred_given_true = np.zeros([n_classes, n_classes])

        predictions = model.predict(X)
        if weights is None:
            weights = np.ones_like(label_vector)

        for cl in range(n_classes):
            sel = np.array(label_vector == cl, dtype=bool)
            true_class_sum = np.sum(weights[sel])
            pred_class_sums = np.bincount(predictions[sel], weights=weights[sel], minlength=n_classes)
            print(cl, pred_class_sums)
            # if there are no true labels for this class, make no adjustment to it
            if true_class_sum == 0:
                self._p_pred_given_true[cl, cl] = 1.0
            else:
                # otherwise, add up the predictions and normalize
                p_pred = pred_class_sums / float(true_class_sum)
                self._p_pred_given_true[:, cl] = p_pred

    def predict_proportions(self, X, weights):
        n_classes = self._model.get_n_classes()
        if n_classes == 2:
            predictions = self._model.predict(X)
            return self.apply_correction(predictions, weights)
        else:
            sys.exit("ACC not yet implemented for more than two classes")

    def apply_correction(self, predictions, weights):
        return self.apply_acc_binary(predictions, weights=weights)

    def apply_acc_binary(self, predictions, weights=None):
        """
        compute the adjusted prediction of proportions based on an ACC correction
            using the simple binary formula, but clip to [0, 1]

        :param predictions: vector of predictions (one per item)
        :return: vector of corrected proportions
        """
        if weights is None:
            weights = np.ones_like(predictions)

        # get predicted label proportions
        n_classes = self._model.get_n_classes()
        pred_prop = compute_proportions_from_predicted_labels(predictions, weights, n_classes=n_classes)

        # apply adjustment
        tpr = self._p_pred_given_true[1, 1]
        tnr = self._p_pred_given_true[0, 0]
        if tpr <= (1-tnr):
            print("**** WARNING: tpr < fnr ****; skipping correction")
            p_1 = pred_prop[1]
        else:
            p_1 = (pred_prop[1] - (1 - tnr)) / (tpr - (1-tnr))

        # clip predicted proportions
        if p_1 > 1:
            p_1 = 1
        if p_1 < 0:
            p_1 = 0

        return np.array([1-p_1, p_1])