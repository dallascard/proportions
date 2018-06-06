import os
import sys
from optparse import OptionParser
from collections import Counter

import numpy as np
import pandas as pd

from util import file_handling as fh


def main():
    usage = "%prog train.csv output_dir"
    parser = OptionParser(usage=usage)
    #parser.add_option('-p', dest='prop', default=1.0,
    #                  help='Use only a random proportion of training data: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    train_file = args[0]
    output_dir = args[1]

    if not os.path.exists(output_dir):
        sys.exit("Output dir does not exist")

    #prop = float(options.prop)

    print("Loading data")
    df = load_df(train_file)

    n_train, _ = df.shape

    print("{:d} tweets".format(n_train))

    print("Converting to JSON")
    dayofyears = Counter()

    text_list = []
    days_list = []
    label_list = []

    index = list(df.index)

    #n_items = len(index)
    #if prop < 1.0:
    #    subset_size = int(prop * n_items)
    #    subset = np.random.choice(range(n_items), size=subset_size, replace=False)
    #    index = [index[i] for i in subset]
    #    print("Using a random subset of %d tweets" % subset_size)

    count = 0
    for i in index:
        row = df.loc[i]
        days_list.append(row.date.dayofyear)
        # neg = 0; pos = 4
        pos = int(row.label) // 4
        label_list.append([1-pos, pos])
        text_list.append({'text': str(row.text), 'id': count, 'positive': int(pos), 'dayofyear': int(row.date.dayofyear)})
        dayofyears.update([row.date.dayofyear])
        count += 1

    n_tweets = count
    print("{:d} tweest found".format(n_tweets))

    fh.write_jsonlist(text_list, os.path.join(output_dir, 'all.jsonlist'))

    labels_df = pd.DataFrame(np.vstack(label_list), index=np.arange(n_tweets), columns=['Negative', 'Positive'])
    labels_df.to_csv(os.path.join(output_dir, 'all.labels.csv'))
    print("Label means:", labels_df.mean(axis=0))

    days_df = pd.DataFrame(days_list, index=np.arange(n_tweets), columns=['dayofyear'])
    days_df.to_csv(os.path.join(output_dir, 'all.dayofyear.csv'))
    print("Day of year counts:")
    keys = list(dayofyears.keys())
    keys.sort()
    for k in keys:
        print(k, dayofyears[k])


def load_df(filename):
    df = pd.read_csv(filename, index_col=None, header=None, encoding='Windows-1252')
    print(df.head())
    cols = ['label', 'id', 'date_string', 'query', 'user', 'text']
    df.columns = cols
    print("Converting dates")
    df['date'] = [pd.Timestamp(d) for d in df.date_string]
    return df


if __name__ == '__main__':
    main()
