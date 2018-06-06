import os
import sys
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.externals import joblib
from sklearn.preprocessing import binarize

from models.evaluation import f1_score, acc_score, calibration_score_binary
from models.evaluation import compute_proportions_from_predicted_labels
from models.logistic_regression import LogisticRegression
from models.secondary_model_acc import ACC
from models.secondary_model_platt import Platt
from util import file_handling as fh


def main():
    usage = "%prog input_dir data_prefix"
    parser = OptionParser(usage=usage)
    parser.add_option('-o', dest='output_base_dir', default='output',
                      help='Output directory (None=auto): default=%default')
    parser.add_option('--label', dest='label_name', default='labels',
                      help='Label name (will look for file train_prefix.label_name.csv): default=%default')
    parser.add_option('--random_test_prop', dest='random_test_prop', default=None,
                      help='Do a random train/test split, instead of using metadata (see below): default=%default')
    parser.add_option('--metadata', dest='metadata_name', default=None,
                      help='Metadata name (will look for file train_prefix.metadata_name.csv): default=%default')
    parser.add_option('--field', dest='field', default='test',
                      help='Metadata field on which to split for train and test: default=%default')
    parser.add_option('--train_start', dest='train_start', default=0,
                      help='Start of training range: default=%default')
    parser.add_option('--train_end', dest='train_end', default=0,
                      help='End of training range: default=%default')
    parser.add_option('--test_start', dest='test_start', default=1,
                      help='Use training data from before this field value: default=%default')
    parser.add_option('--test_end', dest='test_end', default=1,
                      help='Last field value of test data to use: default=%default')
    parser.add_option('--max_n_train', dest='max_n_train', default=None,
                      help='Limit the train+dev set size to this many items (chosen randomly; None=no limit): default=%default')
    parser.add_option('--sample', action="store_true", dest="sample", default=False,
                      help='Sample labels for training instances with multiple annotations: default=%default')
    parser.add_option('--dev_folds', dest='dev_folds', default=5,
                      help='Split training data into this many folds: default=%default')
    parser.add_option('--dev_fold', dest='dev_fold', default=0,
                      help='Which dev fold to use as a dev set: default=%default')
    parser.add_option('--objective', dest='objective', default='f1',
                      help='Objective for choosing best alpha [f1|acc|calibration]: default=%default')
    parser.add_option('--average', dest='average', default=None,
                      help='average for multi-class eval [macro|micro|...]?: default=%default')
    parser.add_option('--penalty', dest='penalty', default='l1',
                      help='Regularization type: default=%default')
    parser.add_option('--alpha_min', dest='alpha_min', default=0.01,
                      help='Minimum value of training hyperparameter: default=%default')
    parser.add_option('--alpha_max', dest='alpha_max', default=1000,
                      help='Maximum value of training hyperparameter: default=%default')
    parser.add_option('--n_alphas', dest='n_alphas', default=11,
                      help='Number of alpha values to try: default=%default')
    parser.add_option('--cshift', action="store_true", dest="cshift", default=False,
                      help='Use reweighting for covariate shift: default=%default')
    parser.add_option('--max_cshift', dest='max_cshift', default=100000,
                      help='Maximum number of data points to use for covariate shift model: default=%default')
    parser.add_option('--features', dest='features', default='unigrams,bigrams',
                      help='Comma-separated list of features: default=%default')
    parser.add_option('--min_dfs', dest='min_dfs', default='1,2',
                      help='Minimum document frequencies (single or comma-separated): default=%default')
    parser.add_option('--transforms', dest='transforms', default='binarize',
                      help='Feature transforms [binarize|] (single or comma-separated): default=%default')
    parser.add_option('--seed', dest='seed', default=42,
                      help='Random seed (None=random): default=%default')

    (options, args) = parser.parse_args()

    config = {}
    config['input_dir'] = args[0]
    config['prefix'] = args[1]
    config['output_base_dir'] = options.output_base_dir
    config['label_name'] = options.label_name
    config['random_test_prop'] = options.random_test_prop
    config['metadata_name'] = options.metadata_name
    config['field'] = options.field
    config['train_start'] = int(options.train_start)
    config['train_end'] = int(options.train_end)
    config['test_start'] = int(options.test_start)
    config['test_end'] = int(options.test_end)
    if options.max_n_train is None:
        config['max_n_train'] = options.max_n_train
    else:
        config['max_n_train'] = int(options.max_n_train)
    config['sample_labels'] = options.sample
    config['dev_folds'] = options.dev_folds
    config['dev_fold'] = options.dev_fold
    config['objective'] = options.objective
    config['average'] = options.average
    config['penalty'] = options.penalty
    config['n_alphas'] = int(options.n_alphas)
    config['alpha_min'] = float(options.alpha_min)
    config['alpha_max'] = float(options.alpha_max)
    config['cshift'] = options.cshift
    config['max_cshift'] = int(options.max_cshift)
    config['features'] = options.features
    config['min_dfs'] = options.min_dfs
    config['transforms'] = options.transforms
    config['seed'] = options.seed
    if config['seed'] is not None:
        print("Setting random seed to be %d" % int(config['seed']))
        np.random.seed(int(config['seed']))

    run(config)


def run(config):
    input_dir = config['input_dir']
    prefix = config['prefix']
    field = config['field']
    label_name = config['label_name']
    random_test_prop = config['random_test_prop']
    metadata_name = config['metadata_name']
    train_start = config['train_start']
    train_end = config['train_end']
    test_start = config['test_start']
    test_end = config['test_end']
    max_n_train = config['max_n_train']
    sample_labels = config['sample_labels']
    penalty = config['penalty']
    objective = config['objective']
    average = config['average']
    cshift = config['cshift']

    # make the output directory and save the config file
    output_dir = make_output_dir(config)
    fh.makedirs(output_dir)
    fh.write_to_json(config, os.path.join(output_dir, 'config.json'))

    # load features
    print(input_dir, label_name, train_end, test_start, penalty, objective, cshift)

    print("Loading features")
    all_X, all_ids, all_vocab = load_all_features(input_dir, prefix, config)
    all_ids_index = dict(zip(all_ids, range(len(all_ids))))
    n_items, n_features = all_X.shape
    print("Full feature matrix shape = ", all_X.shape)

    # if desired, do a random split into test and nontest data
    if random_test_prop is not None:
        print("Doing random train/test split")
        test_prop = float(random_test_prop)
        n_test_all = int(n_items * test_prop)
        test_indices = np.random.choice(np.arange(n_items), size=n_test_all, replace=False)
        test_items_all = [all_ids[i] for i in test_indices]
        nontest_items_all = list(set(all_ids) - set(test_items_all))
        n_nontest_all = len(nontest_items_all)

    # alternatively, if metadata exists, use it to split into test and nontest
    elif metadata_name is not None:
        metadata_file = os.path.join(input_dir, prefix + '.' + metadata_name + '.csv')
        metadata_df = pd.read_csv(metadata_file, header=0, index_col=0)
        metadata_df.index = [str(i) for i in metadata_df.index]
        field_vals = list(set(metadata_df[field].values))
        field_vals.sort()
        print("Splitting data according to %s" % field)
        print("Values:", field_vals)

        print("Testing on %s to %s" % (test_start, test_end))
        # first, split into training and non-train data based on the field of interest
        test_selector_all = (metadata_df[field] >= test_start) & (metadata_df[field] <= test_end)
        metadata_test_df = metadata_df[test_selector_all]
        test_items_all = list(metadata_test_df.index)
        n_test_all = len(test_items_all)

        nontest_selector_all = (metadata_df[field] >= train_start) & (metadata_df[field] <= train_end)
        metadata_nontest_df = metadata_df[nontest_selector_all]
        nontest_items_all = list(metadata_nontest_df.index)
        n_nontest_all = len(nontest_items_all)

    # otherwise, there is not test data; just train a model
    else:
        nontest_items_all = list(all_ids)
        n_nontest_all = len(nontest_items_all)
        test_items_all = []
        n_test_all = 0

    # if there is test data, learn a model to distinguish train from test (if desired):
    weights_df = pd.DataFrame(np.ones(len(all_ids)), index=all_ids, columns=['weight'])
    if n_test_all > 0 and cshift:
        print("Training models for covariates shift")
        # split test and nontest to get balanced subsets
        test_items_1 = list(np.random.choice(test_items_all, size=int(n_test_all/2), replace=False))
        test_items_2 = list(set(test_items_all) - set(test_items_1))
        nontest_items_1 = list(np.random.choice(nontest_items_all, size=int(n_nontest_all/2), replace=False))
        nontest_items_2 = list(set(nontest_items_all) - set(nontest_items_1))

        # combine the test and nontest data into two balanced sets
        cset1_items = nontest_items_1 + test_items_1
        cset2_items = nontest_items_2 + test_items_2
        y1 = [0] * len(test_items_1) + [1] * len(nontest_items_1)
        y2 = [0] * len(test_items_2) + [1] * len(nontest_items_2)

        cset1_indices = [all_ids_index[i] for i in cset1_items]
        cset2_indices = [all_ids_index[i] for i in cset2_items]
        X1 = all_X[cset1_indices, :]
        X2 = all_X[cset2_indices, :]

        # train two models, one on each half of the data, using the other as a dev set
        cshift_model1 = LogisticRegression(n_classes=2, penalty='l2', objective='acc')
        cshift_model1.create_alpha_grid(config['n_alphas'], config['alpha_min'], config['alpha_max'])
        cshift_model1.fit(X1, y1, None, X2, y2, None, 1)

        cshift_model2 = LogisticRegression(n_classes=2, penalty='l2', objective='acc')
        cshift_model2.create_alpha_grid(config['n_alphas'], config['alpha_min'], config['alpha_max'])
        cshift_model2.fit(X2, y2, None, X1, y1, None, 1)

        # now get the models' predictions on the dev data, which will inform future weighting
        y1_pred_probs = cshift_model2.predict_proba(X1)
        for i, item in enumerate(nontest_items_1):
            weights_df.loc[item] = n_nontest_all / float(n_test_all) * (1.0/y1_pred_probs[i, 0] - 1)

        y2_pred_probs = cshift_model1.predict_proba(X2)
        for i, item in enumerate(nontest_items_1):
            weights_df.loc[item] = n_nontest_all / float(n_test_all) * (1.0/y2_pred_probs[i, 0] - 1)

    # reset random seed for consistency with/without using cshift
    if config['seed'] is not None:
        np.random.seed(int(config['seed']))

    print("Weights mean/min/max:", np.mean(weights_df.values), np.min(weights_df.values), np.max(weights_df.values))

    # only keep the items in the train and test sets
    all_items = nontest_items_all + test_items_all
    print("Train: %d, Test: %d (labeled and unlabeled)" % (n_nontest_all, n_test_all))

    # load labels
    label_file = os.path.join(input_dir, prefix + '.' + label_name + '.csv')
    labels_df = pd.read_csv(label_file, index_col=0, header=0)
    labels_df.index = [str(i) for i in labels_df.index]
    labels_df = labels_df.loc[all_items]
    class_names = labels_df.columns

    # find the labeled items
    print("Subsetting items with labels")
    label_sums_df = labels_df.sum(axis=1)
    labeled_item_selector = label_sums_df > 0
    labels_df = labels_df[labeled_item_selector]
    n_labeled_items, n_classes = labels_df.shape
    print("%d labeled items and %d classes" % (n_labeled_items, n_classes))
    labeled_items = set(labels_df.index)

    if n_classes > 2 and config['objective'] == 'calibration':
        sys.exit("*ERROR*: Calibration objective has not been implemented for more than 2 classes")

    nontest_items = [i for i in nontest_items_all if i in labeled_items]
    test_items = [i for i in test_items_all if i in labeled_items]
    n_nontest = len(nontest_items)
    n_test = len(test_items)

    # take a subset of the nontest items up to a max size, if desired.
    if max_n_train is not None and n_nontest_all > max_n_train:
        print("Sampling a set of %d labels" % max_n_train)
        nontest_indices = np.random.choice(np.arange(n_nontest_all), size=max_n_train, replace=False)
        nontest_items = [nontest_items[i] for i in nontest_indices]
        n_nontest = len(nontest_items)

    # split the training set into train and dev
    print("Splitting nontest into train and dev")
    np.random.shuffle(nontest_items)
    n_dev = int(n_nontest / config['dev_folds'])
    dev_fold = int(config['dev_fold'])
    dev_items = nontest_items[n_dev * dev_fold: n_dev * (dev_fold+1)]
    train_items = list(set(nontest_items) - set(dev_items))
    train_items.sort()
    dev_items.sort()
    n_train = len(train_items)
    n_dev = len(dev_items)

    print("Train: %d, dev: %d, test: %d" % (n_train, n_dev, n_test))
    fh.write_list_to_text([str(n_train)], os.path.join(output_dir, 'train.n.txt'))
    fh.write_list_to_text([str(n_test)], os.path.join(output_dir, 'test.n.txt'))
    fh.write_list_to_text([str(n_dev)], os.path.join(output_dir, 'dev.n.txt'))

    test_labels_df = labels_df.loc[test_items]
    nontest_labels_df = labels_df.loc[nontest_items]
    train_labels_df = labels_df.loc[train_items]
    dev_labels_df = labels_df.loc[dev_items]

    test_weights_df = weights_df.loc[test_items]
    nontest_weights_df = weights_df.loc[nontest_items]
    train_weights_df = weights_df.loc[train_items]
    dev_weights_df = weights_df.loc[dev_items]

    # Convert (possibly multiply-annotated) labels to one label per instance, either by duplicating or sampling
    test_labels_df, test_weights_df = prepare_labels(test_labels_df, sample=False, weights_df=test_weights_df)
    nontest_labels_df, nontest_weights_df = prepare_labels(nontest_labels_df, sample=sample_labels, weights_df=nontest_weights_df)
    train_labels_df, train_weights_df = prepare_labels(train_labels_df, sample=sample_labels, weights_df=train_weights_df)
    dev_labels_df, dev_weights_df = prepare_labels(dev_labels_df, sample=sample_labels, weights_df=dev_weights_df)

    test_labels_df.to_csv(os.path.join(output_dir, 'test_labels.csv'))
    nontest_labels_df.to_csv(os.path.join(output_dir, 'nontest_labels.csv'))
    train_labels_df.to_csv(os.path.join(output_dir, 'train_labels.csv'))
    dev_labels_df.to_csv(os.path.join(output_dir, 'dev_labels.csv'))

    test_weights_df.to_csv(os.path.join(output_dir, 'test_weights.csv'))
    nontest_weights_df.to_csv(os.path.join(output_dir, 'nontest_weights.csv'))
    train_weights_df.to_csv(os.path.join(output_dir, 'train_weights.csv'))
    dev_weights_df.to_csv(os.path.join(output_dir, 'dev_weights.csv'))

    # get one-row-hot label matrices for each subset
    train_labels = train_labels_df.values
    dev_labels = dev_labels_df.values
    test_labels = test_labels_df.values
    nontest_labels = nontest_labels_df.values

    # get weight vectors for each subset
    train_weights = train_weights_df.values[:, 0]
    dev_weights = dev_weights_df.values[:, 0]
    test_weights = test_weights_df.values[:, 0]
    nontest_weights = nontest_weights_df.values[:, 0]

    # get new item lists which correspond to the label data frames
    test_items = list(test_labels_df.index)
    dev_items = list(dev_labels_df.index)
    train_items = list(train_labels_df.index)

    n_test = len(test_items)

    # gather training features
    feature_index = dict(zip(all_ids, range(len(all_ids))))
    train_indices = [feature_index[i] for i in train_items]
    dev_indices = [feature_index[i] for i in dev_items]
    test_indices = [feature_index[i] for i in test_items]

    train_X = all_X[train_indices, :]
    dev_X = all_X[dev_indices, :]
    test_X = all_X[test_indices, :]

    print(train_X.shape, dev_X.shape, test_X.shape)

    nontest_prop = np.dot(nontest_weights, nontest_labels) / nontest_weights.sum()
    print("Non-test label proportions:", nontest_prop)
    fh.write_list_to_text([str(nontest_prop[1])], os.path.join(output_dir, 'nontest.prop.txt'))

    if n_test > 0:
        test_prop = np.dot(test_weights, test_labels) / test_weights.sum()
        print("Test label proportions:", test_prop)
        fh.write_list_to_text([str(test_prop[1])], os.path.join(output_dir, 'test.prop.true.txt'))
        fh.write_list_to_text([str(np.abs(test_prop[1] - nontest_prop[1]))], os.path.join(output_dir, 'test.prop.ae.nontest.txt'))
    else:
        test_prop = None

    pos_label = 1
    # use zero as the positive label if it the minority class
    if n_classes == 2:
        if nontest_prop[1] > 0.5:
            pos_label = 0
            print("Using %d as the positive label" % pos_label)

    # convert the label matrices into a categorical label vector
    train_label_vector = np.argmax(train_labels, axis=1)
    test_label_vector = np.argmax(test_labels, axis=1)
    dev_label_vector = np.argmax(dev_labels, axis=1)

    # train a model
    model = LogisticRegression(n_classes=n_classes, penalty=penalty, objective=objective)
    model.create_alpha_grid(config['n_alphas'], config['alpha_min'], config['alpha_max'])
    model.fit(train_X, train_label_vector, train_weights, dev_X, dev_label_vector, dev_weights, pos_label, average)

    print("Number of non-zero weights = %d" % model.get_model_size())

    # predict on train, dev, and test data
    train_f1, train_acc, train_cal = predict_evaluate_and_save(model, train_X, train_items, class_names, train_label_vector, pos_label=pos_label, average=average, weights=train_weights, output_dir=output_dir, output_prefix='train')
    dev_f1, dev_acc, dev_cal = predict_evaluate_and_save(model, dev_X, dev_items, class_names, dev_label_vector, pos_label=pos_label, average=average, weights=dev_weights, output_dir=output_dir, output_prefix='dev')
    if n_test > 0:
        test_f1, test_acc, test_cal = predict_evaluate_and_save(model, test_X, test_items, class_names, test_label_vector, pos_label=pos_label, average=average, weights=test_weights, output_dir=output_dir, output_prefix='test')

    else:
        test_f1 = np.nan
        test_acc = np.nan
        test_cal = np.nan
    print("Accuracy values: train %0.4f; dev %0.4f; test %0.4f" % (train_acc, dev_acc, test_acc))
    print("F1 values: train %0.4f; dev %0.4f; test %0.4f" % (train_f1, dev_f1, test_f1))
    #print("Cal values: train %0.4f; dev %0.4f; test %0.4f" % (train_cal, dev_cal, test_cal))

    if n_test > 0:
        test_pred = model.predict(test_X)
        cc_prop = compute_proportions_from_predicted_labels(test_pred, test_weights, n_classes=2)
        print("Predicted proportions on test:")
        print("CC :", cc_prop)
        fh.write_list_to_text([str(cc_prop[1])], os.path.join(output_dir, 'test.prop.cc.txt'))
        fh.write_list_to_text([str(np.abs(test_prop[1] - cc_prop[1]))], os.path.join(output_dir, 'test.prop.ae.cc.txt'))
        test_pred_probs = model.predict_proba(test_X)
        pcc_prop = np.dot(test_weights, test_pred_probs) / np.sum(test_weights)
        print("PCC:", pcc_prop)
        fh.write_list_to_text([str(pcc_prop[1])], os.path.join(output_dir, 'test.prop.pcc.txt'))
        fh.write_list_to_text([str(np.abs(test_prop[1] - pcc_prop[1]))], os.path.join(output_dir, 'test.prop.ae.pcc.txt'))

    if n_test > 0:
        # create a secondary ACC model
        print("Fitting ACC")
        acc_model = ACC()
        acc_model.fit(model, dev_X, dev_label_vector, dev_weights)
        acc_proportions = acc_model.predict_proportions(test_X, test_weights)
        print("ACC proportions:", acc_proportions)
        fh.write_list_to_text([str(acc_proportions[1])], os.path.join(output_dir, 'test.prop.acc.txt'))
        fh.write_list_to_text([str(np.abs(test_prop[1] - acc_proportions[1]))], os.path.join(output_dir, 'test.prop.ae.acc.txt'))

        # create a secondary calibration model
        print("Fitting Platt")
        platt_model = Platt()
        platt_model.fit(model, dev_X, dev_label_vector, dev_weights, smoothing=True)
        platt_proportions = platt_model.predict_proportions(test_X, test_weights)
        print("Platt proportions:", platt_proportions)
        fh.write_list_to_text([str(platt_proportions[1])], os.path.join(output_dir, 'test.prop.platt.txt'))
        fh.write_list_to_text([str(np.abs(test_prop[1] - platt_proportions[1]))], os.path.join(output_dir, 'test.prop.ae.platt.txt'))

    print_top_words(model, dev_X, all_vocab, n_classes=n_classes, n_words=40, output_dir=output_dir)
    joblib.dump(model, os.path.join(output_dir, 'model.pkl'))
    #fh.write_list_to_text(all_vocab, os.path.join(output_dir, 'model.vocab.txt.gz'), do_gzip=True)
    fh.write_to_json(all_vocab, os.path.join(output_dir, 'model.vocab.json.test.gz'), sort_keys=False, do_gzip=True)
    #fh.write_to_json(all_vocab, os.path.join(output_dir, 'model.vocab.json'), sort_keys=False)

    print("")


def make_output_dir(config):
    output_dir = ''
    output_dir += config['label_name']
    if config['random_test_prop'] is None:
        output_dir += '_' + config['field']
        output_dir += '_' + str(config['train_start']) + '-' + str(config['train_end'])
        output_dir += '_' + str(config['test_start']) + '-' + str(config['test_end'])
    else:
        output_dir += '_' + 'rand' + str(config['random_test_prop'])
    output_dir += '_' + 'max' + str(config['max_n_train'])
    output_dir += '_' + 's' + str(int(config['sample_labels']))
    output_dir += '_' + 'dev' + str(config['dev_fold']) + 'of' + str(config['dev_folds'])
    output_dir += '_' + config['objective']
    output_dir += '_' + config['penalty']
    output_dir += '_' + 'cshift' + str(int(config['cshift']))
    output_dir += '_' + config['features']
    output_dir += '_' + config['min_dfs']
    output_dir += '_' + config['transforms']
    if config['seed'] is not None:
        output_dir += '_s' + str(config['seed'])
    else:
        output_dir += '_sNone'

    output_dir = os.path.join(config['input_dir'], config['output_base_dir'], output_dir)
    return output_dir


def load_all_features(input_dir, prefix, config):
    all_vocab = []
    features = config['features']
    min_dfs = config['min_dfs']
    transforms = config['transforms']
    features = features.split(',')
    n_features = len(features)
    min_dfs = min_dfs.split(',')
    transforms = transforms.split(',')
    if len(min_dfs) == 1:
        min_dfs = min_dfs * n_features
    if len(transforms) == 1:
        transforms = transforms * n_features
    assert len(min_dfs) == n_features
    assert len(transforms) == n_features

    feature_matrices = []

    all_ids = None
    for f_i, feature in enumerate(features):
        counts, ids, vocab = load_feature(input_dir, prefix, feature, min_df=int(min_dfs[f_i]), transform=transforms[f_i])
        feature_matrices.append(counts)
        all_vocab.extend(vocab)
        if all_ids is None:
            all_ids = ids
        else:
            assert all_ids == ids
    if len(feature_matrices) > 1:
        all_X = sparse.hstack(feature_matrices)
    else:
        all_X = feature_matrices[0]

    return all_X, all_ids, all_vocab


def load_feature(input_dir, file_prefix, feature_name, min_df=1, transform='binarize'):
    npz_file = os.path.join(input_dir, file_prefix + '.' + feature_name + '.npz')
    ids_file = os.path.join(input_dir, file_prefix + '.' + feature_name + '.ids.json')
    vocab_file = os.path.join(input_dir, file_prefix + '.' + feature_name + '.vocab.json')

    X = fh.load_sparse(npz_file)
    ids = fh.read_json(ids_file)
    vocab = fh.read_json(vocab_file)

    ids = [str(i) for i in ids]

    n_items, n_features = X.shape
    binarized = binarize(X)
    col_sums = np.array(binarized.sum(axis=0)).reshape((n_features, ))
    col_sel = np.array(col_sums >= min_df)

    print("Doing column selection")
    if transform == 'binarize':
        print("Binarizing")
        X = binarized[:, col_sel]
    else:
        X = X[:, col_sel]
    vocab = [vocab[i] for i in range(n_features) if col_sel[i]]

    return X, ids, vocab


def prepare_labels(labels_df, sample=False, weights_df=None):
    """
    Deal with multiple annotations of binary labels.
    Either convert to multiple labels per item, with corresponding weights
    Or sample one label per item.
    """
    items = list(labels_df.index)
    n_items, n_classes = labels_df.shape
    row_sums = np.array(labels_df.values.sum(axis=1).reshape((n_items, 1)))

    if weights_df is None:
        weights = np.ones(n_items)
    else:
        weights = weights_df.values.reshape((n_items, ))

    if sample:
        print("Sampling labels")
        # normalize the labels
        temp = labels_df.values / row_sums
        samples = np.zeros([n_items, n_classes], dtype=int)
        for i in range(n_items):
            index = np.random.choice(np.arange(n_classes), size=1, p=temp[i, :])
            samples[i, index] = 1
        labels_df = pd.DataFrame(samples, index=labels_df.index, columns=labels_df.columns)
        weights_df = pd.DataFrame(weights, index=labels_df.index, columns=['weight'])

    else:
        all_labels = []
        all_weights = []
        all_items = []
        values = labels_df.values / row_sums
        # duplicate all labels for each class, weighted proportionally
        for c in range(n_classes):
            #values = labels_df.values[:, c].reshape((n_items, ))
            class_weights = values[:, c] * weights
            class_labels = np.zeros([n_items, n_classes], dtype=int)
            class_labels[:, c] = 1
            all_labels.append(class_labels)
            all_weights.extend(class_weights)
            all_items.extend(items)

        items = all_items
        values = np.vstack(all_labels)
        weights = np.array(all_weights)

        # only keep those rows with positive weights
        item_sel = weights > 0.0
        values = values[item_sel, :]
        items = [items[i] for i in range(n_items * n_classes) if item_sel[i]]
        weights = [weights[i] for i in range(n_items * n_classes) if item_sel[i]]

        labels_df = pd.DataFrame(values, index=items, columns=labels_df.columns)
        weights_df = pd.DataFrame(weights, index=items, columns=['weight'])

    return labels_df, weights_df


def predict_evaluate_and_save(model, X, items, class_names, label_vector, pos_label=1, average=None, weights=None, output_dir=None, output_prefix='train'):
    n_classes = len(class_names)
    predictions = model.predict(X)
    pred_probs = model.predict_proba(X)
    f1 = f1_score(label_vector, predictions, n_classes=n_classes, pos_label=pos_label, average=average, weights=weights)
    acc = acc_score(label_vector, predictions, weights=weights)

    if n_classes == 2:
        cal = calibration_score_binary(label_vector, pred_probs, weights)
        if output_dir is not None:
            fh.write_list_to_text([str(cal)], os.path.join(output_dir, output_prefix + '.cal.txt'))
    else:
        cal = None

    if output_dir is not None:
        fh.write_list_to_text([str(f1)], os.path.join(output_dir, output_prefix + '.f1.txt'))
        fh.write_list_to_text([str(acc)], os.path.join(output_dir, output_prefix + '.acc.txt'))
        pred_df = pd.DataFrame(predictions, index=items, columns=['pred'])
        pred_df.to_csv(os.path.join(output_dir, output_prefix + '.pred.csv'))
        pred_prob_df = pd.DataFrame(pred_probs, index=items, columns=class_names)
        pred_prob_df.to_csv(os.path.join(output_dir, output_prefix + '.pred_prob.csv'))

    return f1, acc, cal


def print_top_words(model, counts, vocab, n_classes=2, n_words=20, output_dir=None):
    n_items, vocab_size = counts.shape
    if n_classes == 2:
        coefs = model.get_coefs()
        if coefs is not None:
            coefs = coefs.reshape((vocab_size, ))
        else:
            coefs = np.zeros(vocab_size)
        order = list(np.argsort(coefs))
        neg_features = [vocab[i] for i in order[:n_words] if -coefs[i] > 1e-5]
        order.reverse()
        pos_features = [vocab[i] for i in order[:n_words] if coefs[i] > 1e-5]
        print("Model weights")
        print("Positive features:", ' '.join(pos_features))
        print("Negative features:", ' '.join(neg_features))

        word_counts = np.array(counts.sum(axis=0)).reshape((len(coefs), ))
        importance = coefs * word_counts / float(n_items)
        order = list(np.argsort(importance))
        neg_features = [vocab[i] for i in order[:n_words] if -coefs[i] > 1e-5]
        order.reverse()
        pos_features = [vocab[i] for i in order[:n_words] if coefs[i] > 1e-5]
        print("Weighted by counts")
        print("Positive features:", ' '.join(pos_features))
        print("Negative features:", ' '.join(neg_features))

        if output_dir is not None:
            df = pd.DataFrame(np.zeros([len(vocab), 2]), index=vocab, columns=['weight', 'importance'])
            df['weight'] = coefs
            df['importance'] = importance
            df.to_csv(os.path.join(output_dir, 'model.csv'))

    else:
        word_counts = np.array(counts.sum(axis=0)).reshape((vocab_size, ))
        print("Model weights")
        for c in range(n_classes):
            coefs = model.get_coefs(target_class=c)
            if coefs is not None:
                coefs = coefs.reshape((vocab_size, ))
            else:
                coefs = np.zeros(vocab_size)
            order = list(np.argsort(coefs))
            order.reverse()
            pos_features = [vocab[i] for i in order[:n_words] if coefs[i] > 1e-5]
            print(c, ' '.join(pos_features))

        print("Weighted by counts")
        for c in range(n_classes):
            coefs = model.get_coefs(target_class=c)
            if coefs is not None:
                coefs = coefs.reshape((vocab_size, ))
            else:
                coefs = np.zeros(vocab_size)
            importance = coefs * word_counts / float(n_items)
            order = list(np.argsort(importance))
            order.reverse()
            pos_features = [vocab[i] for i in order[:n_words] if coefs[i] > 1e-5]
            print(c, ' '.join(pos_features))

        if output_dir is not None:
            for c in range(n_classes):
                coefs = model.get_coefs(target_class=c)
                if coefs is not None:
                    coefs = coefs.reshape((vocab_size, ))
                else:
                    coefs = np.zeros(vocab_size)
                importance = coefs * word_counts / float(n_items)
                df = pd.DataFrame(np.zeros([len(vocab), 2]), index=vocab, columns=['weight', 'importance'])
                df['weight'] = coefs
                df['importance'] = importance
                df.to_csv(os.path.join(output_dir, 'model' + str(c) + '.csv'))


if __name__ == '__main__':
    main()
