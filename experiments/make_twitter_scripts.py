import os
from optparse import OptionParser


def main():
    usage = "%prog"
    parser = OptionParser(usage=usage)
    parser.add_option('--omp', dest='omp', default=None,
                      help='Maximum number of threads: default=%default')
    #parser.add_option('-n', dest='n', default=20,
    #                  help='Number of words to print: default=%default')
    #parser.add_option('--sort_by', dest='sort_by', default=None,
    #                  help='Print the output of a column: default=%default')
    #parser.add_option('--omp', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()
    omp_num_threads = options.omp
    os.makedirs('scripts')
    make_scripts(omp_num_threads)


def make_scripts(omp_num_threads=None):

    days = [108, 109, 110, 121, 122, 123, 129, 130, 133, 136, 137, 148, 149, 150, 151, 152, 153, 154, 155, 156, 157, 165]
    for penalty in ['l1', 'l2']:
        for day in days:
            input_dir = os.path.join('data', 'twitter')
            outfile = os.path.join('scripts', 'twitter_' + str(day) + '_' + penalty + '.sh')
            with open(outfile, 'w') as f:
                if omp_num_threads is not None:
                    base_line = 'OMP_NUM_THREADS=' + omp_num_threads + ' '
                else:
                    base_line = ''
                base_line += 'python run_lr_grid.py ' + input_dir + ' all ' + ' --label labels'
                base_line += ' --sample --features unigrams,bigrams --min_dfs 0,0 --penalty ' + penalty
                base_line += ' --metadata dayofyear --field dayofyear --train_start ' + str(day) + ' --train_end ' + str(day)
                base_line += ' --test_start ' + str(day+1) + ' --test_end ' + str(day+1)
                for objective in ['f1', 'calibration']:
                    for max_n_train in ['500', '4000']:
                        for dev_fold in range(1, 5):
                            line = base_line + ' --objective ' + objective + ' --max_n_train ' + max_n_train + ' --dev_fold ' + str(dev_fold)
                            f.write(line + '\n')
                            if objective == 'f1':
                                f.write(line + ' --cshift' + '\n')


if __name__ == '__main__':
    main()