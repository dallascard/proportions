import sys

import numpy as np
from scipy import sparse

from models.logistic_regression import LogisticRegression


class Platt:
    """
    Apply platt scaling to a score classifier
    """

    def __init__(self, penalty='l2', alpha=100000.0):
        self._penalty = penalty
        self._alpha = alpha
        self._p_pred_given_true = None
        self._base_model = None
        self._platt_model = None

    def fit(self, model, X, label_vector, weights, smoothing=False):
        n_classes = model.get_n_classes()
        if n_classes > 2:
            sys.exit("Platt scaling not yet implemented for more than 2 classes.")

        self._base_model = model

        if smoothing:
            X, label_vector, weights = self.reweight_data(X, label_vector, weights)

        scores = np.reshape(model.score(X), (len(label_vector), 1))

        bincount = np.bincount(label_vector, minlength=n_classes)
        most_common = np.argmax(bincount)

        # check to see if there is only one label in the training data:
        if bincount[most_common] == len(label_vector):
            print("Only label %d found in dev data; skipping Platt" % most_common)
        else:
            self._platt_model = LogisticRegression(n_classes, alpha=self._alpha, penalty=self._penalty, objective='acc')
            self._platt_model.fit(scores, label_vector, weights)

    def predict_proba(self, X):
        if self._platt_model is None:
            return self._base_model.predict_proba(X)
        else:
            scores = self._base_model.score(X)
            scores = scores.reshape((len(scores), 1))
            return self._platt_model.predict_proba(scores)

    def predict(self, X):
        pred_probs = self.predict_proba(X)
        predictions = np.argmax(pred_probs, axis=1)
        return predictions

    def predict_proportions(self, X, weights):
        pred_probs = self.predict_proba(X)
        return np.dot(weights, pred_probs) / np.sum(weights)

    def reweight_data(self, X, label_vector, instance_weights):
        n_classes = self._base_model.get_n_classes()
        cl_sums = np.zeros(n_classes)
        for cl in range(n_classes):
            sel = np.array(label_vector == cl, dtype=bool)
            cl_sums[cl] = np.sum(instance_weights[sel])

        pos_weight = (cl_sums[1] + 1) / float(cl_sums[1] + 2)
        neg_weight = (cl_sums[0] + 1) / float(cl_sums[0] + 2)
        weight_vector = (label_vector * pos_weight + (1-label_vector) * neg_weight) * instance_weights

        if type(X) == list:
            X = X + X
        else:
            X = sparse.vstack([X, X])
        y = np.r_[label_vector, 1-label_vector]
        w = np.r_[weight_vector, 1-weight_vector]

        return X, y, w


