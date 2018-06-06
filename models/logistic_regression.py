import numpy as np
from sklearn.linear_model import LogisticRegression as sklearnLogisticRegression

from models.evaluation import evaluate


class LogisticRegression:
    """
    Wrapper for LogisticRegression with some extra functionality:
    - allow for a default prediction when only one training class if given
    - return the full matrix of probabilities/coefficients even when certain classes are missing from training data
    """

    def __init__(self, n_classes, alpha=1.0, penalty='l2', objective='f1'):
        """
        Create a model
        :param alpha: default regularization strength; override with set_alpha_values or create_alpha_grid
        :param penalty: regularization type
        :param objective: objective [f1|acc|calibration]
        """
        self._model_type = 'LogisticRegression'
        self._n_classes = n_classes
        self._alpha = alpha
        self._alpha_values = [alpha]
        self._penalty = penalty
        self._objective = objective
        self._class_names = None
        self._vocab = None
        self._model = None
        self._default_prediction = None

    def set_alpha_values(self, values):
        """Provide a list of regularization strengths to try in cross validation"""
        self._alpha_values = values

    def create_alpha_grid(self, n_alphas, alpha_min, alpha_max, random=False):
        """Create a grid of regularization strengths to try in cross validation"""
        # create a grid of alpha values evenly spaced in log-space
        if random:
            log_alpha_min = np.log(alpha_min)
            log_alpha_max = np.log(alpha_max)
            log_alpha_values = np.random.rand(n_alphas) * np.log(alpha_max) + (log_alpha_max - log_alpha_min) + log_alpha_min
            self._alpha_values = np.exp(log_alpha_values)
        else:
            alpha_factor = np.power(alpha_max / alpha_min, 1.0/(n_alphas-1))
            self._alpha_values = np.array(alpha_min * np.power(alpha_factor, np.arange(n_alphas)))

    def fit(self, train_X, train_labels, train_weights=None, dev_X=None, dev_labels=None, dev_weights=None, pos_label=1, average='micro'):
        """
        Fit a logistic regression model to data
        If dev data is given, and a set of alpha values have been set, will do cross validation based on the given objective
        :param train_X: a (sparse) matrix of values
        :param train_labels: a vector of categorical labels
        :param train_weights: a vector of instance weights
        :param dev_X
        :param dev_labels
        :param dev_weights
        :param pos_label: which label to use as positive when computing f1 for binary labels
        :param average: type of averaging to use when computing f1 for more than two classes
        """
        n_alphas = len(self._alpha_values)
        fold_evals = np.zeros(n_alphas)

        models = []

        bincount = np.bincount(train_labels, minlength=self._n_classes)
        most_common = np.argmax(bincount)

        # check to see if there is only one label in the training data:
        if bincount[most_common] == len(train_labels):
            print("Only label %d found in training data" % most_common)
            self._default_prediction = most_common
            self._model = None
        elif dev_X is not None and dev_labels is not None:
            print("Doing cross validation")
            for i, alpha in enumerate(self._alpha_values):
                model = sklearnLogisticRegression(penalty=self._penalty, C=alpha)
                model.fit(train_X, train_labels, sample_weight=train_weights)
                train_pred = model.predict(train_X)
                train_pred_probs = model.predict_proba(train_X)
                dev_pred = model.predict(dev_X)
                dev_pred_probs = model.predict_proba(dev_X)

                train_eval = evaluate(self._objective, train_labels, train_pred, train_pred_probs, n_classes=self._n_classes, pos_label=pos_label, average=average, weights=train_weights)
                fold_evals[i] = evaluate(self._objective, dev_labels, dev_pred, dev_pred_probs, n_classes=self._n_classes, pos_label=pos_label, average=average, weights=dev_weights)

                print("%d alpha=%0.4f train=%0.4f dev=%0.4f" % (i, alpha, train_eval, fold_evals[i]))
                models.append(model)

            if self._objective == 'calibration':
                best_alpha_index = int(np.argmin(fold_evals))
            else:
                best_alpha_index = int(np.argmax(fold_evals))
            best_alpha = self._alpha_values[best_alpha_index]
            self._model = models[best_alpha_index]
            print("Best alpha = %0.4f; corresponding %s = %0.4f" % (best_alpha, self._objective, fold_evals[best_alpha_index]))
        else:
            print("Training a model with alpha = %0.4f" % self._alpha)
            model = sklearnLogisticRegression(penalty=self._penalty, C=self._alpha)
            model.fit(train_X, train_labels, sample_weight=train_weights)
            train_pred = model.predict(train_X)
            train_pred_probs = model.predict_proba(train_X)
            train_eval = evaluate(self._objective, train_labels, train_pred, train_pred_probs, n_classes=self._n_classes, pos_label=pos_label, average=average, weights=train_weights)
            print("Train %s = %0.4f" % (self._objective, train_eval))
            self._model = model

        return self._alpha_values, fold_evals

    def predict(self, X):
        # if we've stored a default value, then that is our prediction
        if self._model is None:
            # else, get the model to make predictions
            n_items, _ = X.shape
            return np.ones(n_items, dtype=int) * self._default_prediction
        else:
            return self._model.predict(X)

    def predict_proba(self, X):
        n_items, _ = X.shape
        full_probs = np.zeros([n_items, self._n_classes])
        # if we've saved a default label, predict that with 100% confidence
        if self._model is None:
            full_probs[:, self._default_prediction] = 1.0
        else:
            # otherwise, get probabilities from the model
            model_probs = self._model.predict_proba(X)
            # map the classes that were present in the training data back to the full set of classes
            for i, cl in enumerate(self._model.classes_):
                full_probs[:, cl] = model_probs[:, i]
        return full_probs

    def score(self, X):
        if self._model is None:
            n, p = X.shape
            return np.zeros(n)
        else:
            return self._model.decision_function(X)

    def get_coefs(self, target_class=0):
        """
        Return the coefficients for one class
        :param target_class: The class coefficients to retrieve (ignored if self._n_classes = 2)
        :return: A vector of coefficients if the class if found, else None
        """
        if self._model is not None:
            # search for the target class
            for i, cl in enumerate(self._model.classes_):
                if cl == target_class:
                    return self._model.coef_[i]

    def get_intercept(self, target_class=0):
        # if we've saved a default value, there are no intercepts
        intercept = 0
        if self._model is not None:
            # otherwise, see if the model has an intercept for this class
            for i, cl in enumerate(self._model.classes_):
                if cl == target_class:
                    intercept = self._model.intercept_[i]
                    break
        return intercept

    def get_model_size(self):
        if self._model is None:
            return 0
        else:
            coefs = self._model.coef_
            n_nonzero = len(coefs.nonzero()[0])
            return n_nonzero

    def get_n_classes(self):
        return self._n_classes