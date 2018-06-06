import matplotlib as mpl
mpl.use('Agg')
import os
import re
import glob
from optparse import OptionParser

import numpy as np
import pandas as pd

from util import file_handling as fh


def main():
    usage = "%prog base_dir(s) [use __star__ for *]"
    parser = OptionParser(usage=usage)
    parser.add_option('--n_train', dest='n_train', default='500,4000',
                      help='Number of training samples: default=%default')
    parser.add_option('--dev_fold', dest='dev_fold', default=None,
                      help='Dev fold to use (None=all): default=%default')
    parser.add_option('-o', dest='output_prefix', default='test',
                      help='Output prefix: default=%default')
    parser.add_option('--l1', action="store_true", dest="l1", default=False,
                      help='Only look at l1: default=%default')

    (options, args) = parser.parse_args()

    base_dir = args[0]
    base_dir = re.sub(r'__star__', '*', base_dir)

    output_prefix = options.output_prefix
    dev_fold = options.dev_fold
    n_train_vals = options.n_train
    n_train_vals = n_train_vals.split(',')
    if dev_fold is not None:
        dev_string = 'dev' + dev_fold + 'of5'
    else:
        dev_string = 'dev*of5'

    print("Searching", base_dir)

    df = pd.DataFrame()

    if options.l1:
        penalties = ['l1']
    else:
        penalties = ['l1', 'l2']
    results = []
    for penalty in penalties:
        for max_n_train in n_train_vals:
            col = penalty + '_' + max_n_train
            df[col] = 0
            nontest_vals = []
            cc_vals = []
            pcc_vals = []
            acc_vals = []
            platt_vals = []
            cal_vals = []
            cshift_vals = []
            n_files = []

            basename = '*_max' + max_n_train + '_s*_' + dev_string + '_' + 'f1' + '_' + penalty + '_cshift0' + '_*_s42'
            search_string = os.path.join(base_dir, basename, 'test.prop.ae.nontest.txt')
            nontest_files = glob.glob(search_string)
            nontest_files.sort()
            n_files.append(len(nontest_files))
            for f in nontest_files:
                ae = float(fh.read_text(f)[0])
                nontest_vals.append(ae)

            cc_files = glob.glob(os.path.join(base_dir, basename, 'test.prop.ae.cc.txt'))
            cc_files.sort()
            n_files.append(len(cc_files))
            for f in cc_files:
                ae = float(fh.read_text(f)[0])
                cc_vals.append(ae)

            pcc_files = glob.glob(os.path.join(base_dir, basename, 'test.prop.ae.pcc.txt'))
            pcc_files.sort()
            n_files.append(len(pcc_files))
            for f in pcc_files:
                ae = float(fh.read_text(f)[0])
                pcc_vals.append(ae)

            acc_files = glob.glob(os.path.join(base_dir, basename, 'test.prop.ae.acc.txt'))
            acc_files.sort()
            n_files.append(len(acc_files))
            for f in acc_files:
                ae = float(fh.read_text(f)[0])
                acc_vals.append(ae)

            platt_files = glob.glob(os.path.join(base_dir, basename, 'test.prop.ae.platt.txt'))
            platt_files.sort()
            n_files.append(len(platt_files))
            for f in platt_files:
                ae = float(fh.read_text(f)[0])
                platt_vals.append(ae)

            basename = '*_max' + max_n_train + '_s*_' + dev_string + '_' + 'calibration' + '_' + penalty + '_cshift0' + '_*_s42'
            cal_files = glob.glob(os.path.join(base_dir, basename, 'test.prop.ae.pcc.txt'))
            cal_files.sort()
            n_files.append(len(cal_files))
            for f in cal_files:
                ae = float(fh.read_text(f)[0])
                cal_vals.append(ae)

            basename = '*_max' + max_n_train + '_s*_' + dev_string + '_' + 'f1' + '_' + penalty + '_cshift1' + '_*_s42'
            cshift_files = glob.glob(os.path.join(base_dir, basename, 'test.prop.ae.pcc.txt'))
            cshift_files.sort()
            n_files.append(len(cshift_files))
            for f in cshift_files:
                ae = float(fh.read_text(f)[0])
                cshift_vals.append(ae)

            print(n_files)

            df.loc['nontest', col] = np.mean(nontest_vals)
            df.loc['cc', col] = np.mean(cc_vals)
            df.loc['pcc', col] = np.mean(pcc_vals)
            df.loc['acc', col] = np.mean(acc_vals)
            df.loc['cshift', col] = np.mean(cshift_vals)
            df.loc['platt', col] = np.mean(platt_vals)
            df.loc['cal', col] = np.mean(cal_vals)

            results.append([output_prefix, penalty, max_n_train, 'nontest'] + nontest_vals)
            results.append([output_prefix, penalty, max_n_train, 'cc'] + cc_vals)
            results.append([output_prefix, penalty, max_n_train, 'pcc'] + pcc_vals)
            results.append([output_prefix, penalty, max_n_train, 'acc'] + acc_vals)
            results.append([output_prefix, penalty, max_n_train, 'cshift'] + cshift_vals)
            results.append([output_prefix, penalty, max_n_train, 'platt'] + platt_vals)
            results.append([output_prefix, penalty, max_n_train, 'cal'] + cal_vals)


    print(df)

    df_all = pd.DataFrame(columns=['Dataset', 'Penalty', 'n_train', 'method'] + list(range(len(results[0]) - 4)))

    for i in range(len(results)):
        df_all.loc[i] = results[i]

    df_all.to_csv(output_prefix + '.csv')





if __name__ == '__main__':
    main()
