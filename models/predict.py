import os
from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy import sparse
from sklearn.externals import joblib

from run_lr_grid import load_all_features
from util import file_handling as fh


def main():
    usage = "%prog model_dir data_dir data_prefix output_file_probs.csv"
    parser = OptionParser(usage=usage)
    parser.add_option('--metadata', dest='metadata_name', default=None,
                      help='Metadata name (will look for file train_prefix.metadata_name.csv): default=%default')
    parser.add_option('--field', dest='field', default='test',
                      help='Metadata field on which to split for train and test: default=%default')
    parser.add_option('--start', dest='start', default=1,
                      help='Use training data from before this field value: default=%default')
    parser.add_option('--end', dest='end', default=1,
                      help='Last field value of test data to use: default=%default')
    parser.add_option('--features', dest='features', default='n1grams,n2grams',
                      help='Comma-separated list of features: default=%default')
    parser.add_option('--min_dfs', dest='min_dfs', default='1,2',
                      help='Minimum document frequencies (single or comma-separated): default=%default')
    parser.add_option('--transforms', dest='transforms', default='binarize',
                      help='Feature transforms [binarize|] (single or comma-separated): default=%default')

    (options, args) = parser.parse_args()

    config = {}
    config['model_dir'] = args[0]
    config['data_dir'] = args[1]
    config['data_prefix'] = args[2]
    config['output_file'] = args[3]
    config['metadata_name'] = options.metadata_name
    config['field'] = options.field
    config['start'] = int(options.start)
    config['end'] = int(options.end)
    config['features'] = options.features
    config['min_dfs'] = options.min_dfs
    config['transforms'] = options.transforms

    load_and_predict(config)


def load_and_predict(config):
    model_dir = config['model_dir']
    input_dir = config['data_dir']
    prefix = config['data_prefix']
    metadata_name = config['metadata_name']
    field = config['field']
    start = config['start']
    end = config['end']

    print("Loading model")
    model = joblib.load(os.path.join(model_dir, 'model.pkl'))
    model_vocab = fh.read_json(os.path.join(model_dir, 'model.vocab.json'))
    model_vocab_index = dict(zip(model_vocab, range(len(model_vocab))))

    print("Loading features")
    all_X, all_ids, data_vocab = load_all_features(input_dir, prefix, config)
    data_vocab_index = dict(zip(data_vocab, range(len(data_vocab))))
    n_items, n_features = all_X.shape
    print("Full feature matrix shape = ", all_X.shape)

    zeros = np.zeros([len(all_ids), 1])
    all_X = sparse.hstack([all_X, zeros]).tocsc()

    print("Setting vocabulary")
    col_sel = [data_vocab_index[word] if word in data_vocab_index else len(data_vocab) for word in model_vocab]
    all_X = all_X[:, col_sel]

    if metadata_name is not None:
        metadata_file = os.path.join(input_dir, prefix + '.' + metadata_name + '.csv')
        metadata_df = pd.read_csv(metadata_file, header=0, index_col=0)
        metadata_df.index = [str(i) for i in metadata_df.index]
        field_vals = list(set(metadata_df[field].values))
        field_vals.sort()
        print("Splitting data according to %s" % field)
        print("Values:", field_vals)

        print("Testing on %s to %s" % (start, end))
        # first, split into training and non-train data based on the field of interest
        row_selector = (metadata_df[field] >= start) & (metadata_df[field] <= end)
        print(row_selector.shape)
        row_indices = [i for i in range(len(all_ids)) if row_selector[i]]
        all_ids = [id for i, id in enumerate(all_ids) if row_selector[i]]
        print(len(all_ids))
        #all_X = all_X.tocsr()
        print("Selecting rows")
        all_X = all_X[row_indices, :]

    print(all_X.shape)
    X_for_model = all_X

    print("Doing prediction")
    pred = model.predict(X_for_model)
    probs = model.predict_proba(X_for_model)

    df = pd.DataFrame(probs, index=all_ids)
    output_file = config['output_file']
    df.to_csv(output_file)

    proportions = np.bincount(pred) / float(len(pred))
    print(proportions)



if __name__ == '__main__':
    main()
