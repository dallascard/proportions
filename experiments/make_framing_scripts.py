import os
from optparse import OptionParser

from util import file_handling


# TODO: write out all experiments I want to run, and save to bash scripts

FRAMES = ["Economic",
          "Capacity",
          "Morality",
          "Fairness",
          "Legality",
          "Policy",
          "Crime",
          "Security",
          "Health",
          "Quality",
          "Cultural",
          "Public",
          "Political",
          "External",
          "Other"]

os.makedirs('scripts')
#for sample in ['0', '1']:
for subset in ['immigration', 'samesex', 'tobacco']:
    for penalty in ['l1']:
        for max_n_train in ['500', '2000']:
            input_dir = os.path.join('data', 'framing', subset)
            outfile = os.path.join('scripts', 'framing_' + subset + '_' + penalty + '_max' + max_n_train + '.sh')
            with open(outfile, 'w') as f:
                base_line = 'OMP_NUM_THREADS=4 python run_lr_grid.py ' + input_dir + ' all '
                base_line += ' --metadata years --field year --train_start 1990 --train_end 2008 --test_start 2009 --test_end 2012'
                base_line += ' --features unigrams,bigrams --min_dfs 0,0 --penalty ' + penalty
                #base_line += ' --sample '
                for objective in ['f1', 'calibration']:
                    for frame in FRAMES:
                        for dev_fold in range(5):
                            line = base_line + ' --objective ' + objective + ' --max_n_train ' + max_n_train + ' --label ' + frame + ' --dev_fold ' + str(dev_fold)
                            f.write(line + '\n')
                            f.write(line + ' --cshift' + '\n')
