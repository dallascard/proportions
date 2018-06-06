from optparse import OptionParser

import numpy as np
import pandas as pd
from scipy.stats import wilcoxon


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')


    (options, args) = parser.parse_args()

    amazon_df = pd.read_csv('amazon.csv', header=0, index_col=0)
    framing_df = pd.read_csv('framing.csv', header=0, index_col=0)
    yelp_df = pd.read_csv('yelp.csv', header=0, index_col=0)
    twitter_df = pd.read_csv('twitter.csv', header=0, index_col=0)
    datasets = {'framing': framing_df, 'amazon': amazon_df, 'yelp': yelp_df, 'twitter': twitter_df}
    keys = ['framing', 'amazon', 'yelp', 'twitter']


    print("rest vs cal")
    for k in keys:
        df = datasets[k]
        n_train_vals = list(set(df['n_train'].values))
        n_train_vals.sort()
        for penalty in ['l1']:
            for n_train in n_train_vals:
                for method in ['nontest', 'cc', 'pcc', 'acc', 'cshift', 'platt']:
                    vals1 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == method)].values[0, 4:]
                    vals2 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == 'cal')].values[0, 4:]
                    test_result = wilcoxon(vals1, vals2)
                    pval = test_result[1]
                    print(k, penalty, n_train, method, len(vals1), len(vals2), np.mean(vals1) - np.mean(vals2), pval)

    print("worse than cal")
    for k in keys:
        df = datasets[k]
        n_train_vals = list(set(df['n_train'].values))
        n_train_vals.sort()
        for penalty in ['l1']:
            for n_train in n_train_vals:
                for method in ['nontest', 'cc', 'pcc', 'acc', 'cshift', 'platt']:
                    vals1 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == method)].values[0, 4:]
                    vals2 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == 'cal')].values[0, 4:]
                    test_result = wilcoxon(vals1, vals2)
                    pval = test_result[1]
                    if np.mean(vals1) - np.mean(vals2) > 0 and pval < 0.05/48:
                        print(k, penalty, n_train, method, len(vals1), len(vals2), np.mean(vals1) - np.mean(vals2), pval)

    print("better than cc")
    for k in keys:
        df = datasets[k]
        n_train_vals = list(set(df['n_train'].values))
        n_train_vals.sort()
        for penalty in ['l1']:
            for n_train in n_train_vals:
                for method in ['nontest', 'pcc', 'acc', 'cshift', 'platt', 'cal']:
                    vals1 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == method)].values[0, 4:]
                    vals2 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == 'cc')].values[0, 4:]
                    test_result = wilcoxon(vals1, vals2)
                    pval = test_result[1]
                    if np.mean(vals1) - np.mean(vals2) < 0 and pval < 0.05/48:
                        print(k, penalty, n_train, method, len(vals1), len(vals2), np.mean(vals1) - np.mean(vals2), pval)

    print("better than pcc")
    for k in keys:
        df = datasets[k]
        n_train_vals = list(set(df['n_train'].values))
        n_train_vals.sort()
        for penalty in ['l1']:
            for n_train in n_train_vals:
                for method in ['nontest', 'cc', 'acc', 'cshift', 'platt', 'cal']:
                    vals1 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == method)].values[0, 4:]
                    vals2 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == 'cc')].values[0, 4:]
                    test_result = wilcoxon(vals1, vals2)
                    pval = test_result[1]
                    if np.mean(vals1) - np.mean(vals2) < 0 and pval < 0.05/48:
                        print(k, penalty, n_train, method, len(vals1), len(vals2), np.mean(vals1) - np.mean(vals2), pval)

    print("better than acc")
    for k in keys:
        df = datasets[k]
        n_train_vals = list(set(df['n_train'].values))
        n_train_vals.sort()
        for penalty in ['l1']:
            for n_train in n_train_vals:
                for method in ['nontest', 'cc', 'pcc', 'cshift', 'platt', 'cal']:
                    vals1 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == method)].values[0, 4:]
                    vals2 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == 'acc')].values[0, 4:]
                    test_result = wilcoxon(vals1, vals2)
                    pval = test_result[1]
                    if np.mean(vals1) - np.mean(vals2) < 0 and pval < 0.05/48:
                        print(k, penalty, n_train, method, len(vals1), len(vals2), np.mean(vals1) - np.mean(vals2), pval)

    print("worse than ACC")
    for k in keys:
        df = datasets[k]
        n_train_vals = list(set(df['n_train'].values))
        n_train_vals.sort()
        for penalty in ['l1']:
            for n_train in n_train_vals:
                for method in ['nontest', 'cc', 'pcc', 'cshift', 'platt', 'cal']:
                    vals1 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == method)].values[0, 4:]
                    vals2 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == 'acc')].values[0, 4:]
                    test_result = wilcoxon(vals1, vals2)
                    pval = test_result[1]
                    if np.mean(vals1) - np.mean(vals2) > 0 and pval < 0.05/48:
                        print(k, penalty, n_train, method, len(vals1), len(vals2), np.mean(vals1) - np.mean(vals2), pval)

    print("worse than platt")
    for k in keys:
        df = datasets[k]
        n_train_vals = list(set(df['n_train'].values))
        n_train_vals.sort()
        for penalty in ['l1']:
            for n_train in n_train_vals:
                for method in ['nontest', 'cc', 'pcc', 'acc', 'cshift', 'cal']:
                    vals1 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == method)].values[0, 4:]
                    vals2 = df[(df.Penalty == penalty) & (df.n_train == n_train) & (df.method == 'platt')].values[0, 4:]
                    test_result = wilcoxon(vals1, vals2)
                    pval = test_result[1]
                    if np.mean(vals1) - np.mean(vals2) > 0 and pval < 0.05/48:
                        print(k, penalty, n_train, method, len(vals1), len(vals2), np.mean(vals1) - np.mean(vals2), pval)

    print("L1 better than L2")
    for method in ['cc', 'pcc', 'acc', 'cshift', 'platt', 'cal']:
        for k in keys:
            df = datasets[k]
            n_train_vals = list(set(df['n_train'].values))
            n_train_vals.sort()
            for n_train in n_train_vals:
                vals1 = df[(df.Penalty == 'l1') & (df.n_train == n_train) & (df.method == method)].values[0, 4:]
                vals2 = df[(df.Penalty == 'l2') & (df.n_train == n_train) & (df.method == method)].values[0, 4:]
                test_result = wilcoxon(vals1, vals2)
                pval = test_result[1]
                if np.mean(vals1) - np.mean(vals2) < 0 and pval < 0.05/48:
                    print(k, method, n_train, len(vals1), len(vals2), np.mean(vals1) - np.mean(vals2), pval)

    print("L2 better than L1")
    for method in ['cc', 'pcc', 'acc', 'cshift', 'platt', 'cal']:
        for k in keys:
            df = datasets[k]
            n_train_vals = list(set(df['n_train'].values))
            n_train_vals.sort()
            for n_train in n_train_vals:
                vals1 = df[(df.Penalty == 'l1') & (df.n_train == n_train) & (df.method == method)].values[0, 4:]
                vals2 = df[(df.Penalty == 'l2') & (df.n_train == n_train) & (df.method == method)].values[0, 4:]
                test_result = wilcoxon(vals1, vals2)
                pval = test_result[1]
                if np.mean(vals1) - np.mean(vals2) > 0 and pval < 0.05/48:
                    print(k, method, n_train, len(vals1), len(vals2), np.mean(vals1) - np.mean(vals2), pval)


if __name__ == '__main__':
    main()
