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

    pairs = ['Charlotte_Pittsburgh', 'Las_Vegas_Phoenix', 'Tempe_Henderson', 'Toronto_Scottsdale']
    dirs = ['Charlotte', 'Las_Vegas', 'Tempe', 'Toronto']
    for pair_i, pair in enumerate(pairs):
        years = range(2009, 2017)
        for penalty in ['l1', 'l2']:
            for year in years:
                for max_n_train in ['500', '4000']:
                    input_dir = os.path.join('data', 'yelp', dirs[pair_i])
                    outfile = os.path.join('scripts', 'yelp_' + dirs[pair_i] + '_' + str(year) + '_' + penalty + '_' + max_n_train + '.sh')
                    with open(outfile, 'w') as f:
                        if omp_num_threads is not None:
                            base_line = 'OMP_NUM_THREADS=' + omp_num_threads + ' '
                        else:
                            base_line = ''
                        base_line += 'python run_lr_grid.py ' + input_dir + ' --label labels'
                        base_line += ' --sample --features unigrams,bigrams --min_dfs 0,0 --penalty ' + penalty
                        base_line += ' --metadata years --field year --train_start ' + str(year) + ' --train_end ' + str(year)
                        base_line += ' --test_start ' + str(year+1) + ' --test_end ' + str(year+1) + ' --max_n_train ' + max_n_train
                        for objective in ['f1', 'calibration']:
                            for dev_fold in range(0, 5):

                                line = base_line + ' --objective ' + objective + ' ' + pair + ' --dev_fold ' + str(dev_fold)
                                f.write(line + '\n')
                                if objective == 'f1':
                                    f.write(line + ' --cshift' + '\n')


if __name__ == '__main__':
    main()