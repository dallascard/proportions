import os
import re
import glob
from optparse import OptionParser

import numpy as np

from util import file_handling as fh

def main():
    usage = "%prog base_dir"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    base_dir = args[0]

    for dev_fold in range(5):
        dirs = glob.glob(os.path.join(base_dir, '*_dev' + str(dev_fold) + 'of5_*_l1_*'))
        #dirs = glob.glob(os.path.join(base_dir, '*_dev' + str(dev_fold) + 'of5_*'))
        dirs.sort()
        for dir in dirs:
            output_dir = re.sub(r'dev\dof5', 'devAof5', dir)
            print(output_dir)
            if not os.path.exists(output_dir):
                os.makedirs(output_dir)
            test_prop_true = fh.read_text(os.path.join(dir, 'test.prop.true.txt'))[0]
            test_prop_nontest = fh.read_text(os.path.join(dir, 'nontest.prop.txt'))[0]
            test_prop_cc = fh.read_text(os.path.join(dir, 'test.prop.cc.txt'))[0]
            test_prop_pcc = fh.read_text(os.path.join(dir, 'test.prop.pcc.txt'))[0]
            test_prop_acc = fh.read_text(os.path.join(dir, 'test.prop.acc.txt'))[0]
            test_prop_platt = fh.read_text(os.path.join(dir, 'test.prop.platt.txt'))[0]

            if dev_fold == 0:
                with open(os.path.join(output_dir, 'test.prop.true.vals.txt'), 'w') as f:
                    f.write(test_prop_true + ' ')
                with open(os.path.join(output_dir, 'test.prop.nontest.vals.txt'), 'w') as f:
                    f.write(test_prop_nontest + ' ')
                with open(os.path.join(output_dir, 'test.prop.cc.vals.txt'), 'w') as f:
                    f.write(test_prop_cc + ' ')
                with open(os.path.join(output_dir, 'test.prop.pcc.vals.txt'), 'w') as f:
                    f.write(test_prop_pcc + ' ')
                with open(os.path.join(output_dir, 'test.prop.acc.vals.txt'), 'w') as f:
                    f.write(test_prop_acc + ' ')
                with open(os.path.join(output_dir, 'test.prop.platt.vals.txt'), 'w') as f:
                    f.write(test_prop_platt + ' ')
            else:
                with open(os.path.join(output_dir, 'test.prop.true.vals.txt'), 'a') as f:
                    f.write(test_prop_true + ' ')
                with open(os.path.join(output_dir, 'test.prop.nontest.vals.txt'), 'a') as f:
                    f.write(test_prop_nontest + ' ')
                with open(os.path.join(output_dir, 'test.prop.cc.vals.txt'), 'a') as f:
                    f.write(test_prop_cc + ' ')
                with open(os.path.join(output_dir, 'test.prop.pcc.vals.txt'), 'a') as f:
                    f.write(test_prop_pcc + ' ')
                with open(os.path.join(output_dir, 'test.prop.acc.vals.txt'), 'a') as f:
                    f.write(test_prop_acc + ' ')
                with open(os.path.join(output_dir, 'test.prop.platt.vals.txt'), 'a') as f:
                    f.write(test_prop_platt + ' ')

    dirs = glob.glob(os.path.join(base_dir, '*_devAof5_*_l1_*'))
    #dirs = glob.glob(os.path.join(base_dir, '*_devAof5_*'))
    for dir in dirs:
        test_prop_true_vals = fh.read_text(os.path.join(dir, 'test.prop.true.vals.txt'))[0]
        test_prop_true_mean = np.mean([float(v) for v in test_prop_true_vals.split()])
        fh.write_list_to_text([str(test_prop_true_mean)], os.path.join(dir, 'test.prop.true.txt'))

        vals = fh.read_text(os.path.join(dir, 'test.prop.nontest.vals.txt'))[0]
        mean = np.mean([float(v) for v in vals.split()])
        ae = np.abs(test_prop_true_mean - mean)
        fh.write_list_to_text([str(ae)], os.path.join(dir, 'test.prop.ae.nontest.txt'))

        vals = fh.read_text(os.path.join(dir, 'test.prop.cc.vals.txt'))[0]
        mean = np.mean([float(v) for v in vals.split()])
        ae = np.abs(test_prop_true_mean - mean)
        fh.write_list_to_text([str(ae)], os.path.join(dir, 'test.prop.ae.cc.txt'))

        vals = fh.read_text(os.path.join(dir, 'test.prop.pcc.vals.txt'))[0]
        mean = np.mean([float(v) for v in vals.split()])
        ae = np.abs(test_prop_true_mean - mean)
        fh.write_list_to_text([str(ae)], os.path.join(dir, 'test.prop.ae.pcc.txt'))

        vals = fh.read_text(os.path.join(dir, 'test.prop.acc.vals.txt'))[0]
        mean = np.mean([float(v) for v in vals.split()])
        ae = np.abs(test_prop_true_mean - mean)
        fh.write_list_to_text([str(ae)], os.path.join(dir, 'test.prop.ae.acc.txt'))

        vals = fh.read_text(os.path.join(dir, 'test.prop.platt.vals.txt'))[0]
        mean = np.mean([float(v) for v in vals.split()])
        ae = np.abs(test_prop_true_mean - mean)
        fh.write_list_to_text([str(ae)], os.path.join(dir, 'test.prop.ae.platt.txt'))


if __name__ == '__main__':
    main()
