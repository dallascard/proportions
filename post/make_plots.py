from optparse import OptionParser

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sbn


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

    methods = ['nontest', 'cc', 'pcc', 'acc', 'cshift', 'platt', 'cal']
    method_names = ['Train', 'CC', r'PCC$^{\mathrm{F}_1}$', 'ACC', 'Reweighting', 'Platt', 'PCC$^{\mathrm{cal}}$']
    y = list(range(len(method_names)))
    y.reverse()

    # make extinsic plot
    sbn.set(style="white", font_scale=1.5)
    sbn.palplot(sbn.color_palette("hls", 7))

    fig, axes = plt.subplots(1, 4, figsize=(9, 3), sharey=True)
    fig.subplots_adjust(wspace=0)

    add_plot(axes[0], datasets, 'framing', 'l1', 500, methods, method_names, 'MFC (L=500)')
    add_plot(axes[1], datasets, 'framing', 'l1', 2000, methods, method_names, 'MFC (L=2000)')
    add_plot(axes[2], datasets, 'amazon', 'l1', 500, methods, method_names, 'Amazon (L=500)')
    add_plot(axes[3], datasets, 'amazon', 'l1', 4000, methods, method_names, 'Amazon (L=4000)')

    filename = 'extrinsic.pdf'
    plt.savefig(filename, bbox_inches='tight')

    fig, axes = plt.subplots(1, 4, figsize=(9, 3), sharey=True)
    fig.subplots_adjust(wspace=0)

    add_plot(axes[0], datasets, 'yelp', 'l1', 500, methods, method_names, 'Yelp (L=500)')
    add_plot(axes[1], datasets, 'yelp', 'l1', 4000, methods, method_names, 'Yelp (L=4000)')
    add_plot(axes[2], datasets, 'twitter', 'l1', 500, methods, method_names, 'Twitter (L=500)')
    add_plot(axes[3], datasets, 'twitter', 'l1', 4000, methods, method_names, 'Twitter (L=4000)')

    filename = 'intrinsic.pdf'
    plt.savefig(filename, bbox_inches='tight')


    """
    ax.barh([7], [np.mean(nontest_vals)], alpha=0.5)
    ax.scatter(nontest_vals, np.ones_like(nontest_vals)*7, color='k', s=10, alpha=0.5)
    ax.barh([6], [np.mean(cc_vals)], alpha=0.5)
    ax.scatter(cc_vals, np.ones_like(cc_vals)*6, color='k', s=10, alpha=0.5)
    ax.barh([5], [np.mean(pcc_vals)], alpha=0.5)
    ax.scatter(pcc_vals, np.ones_like(pcc_vals)*5, color='k', s=10, alpha=0.5)
    ax.barh([4], [np.mean(acc_vals)], alpha=0.5)
    ax.scatter(acc_vals, np.ones_like(acc_vals)*4, color='k', s=10, alpha=0.5)
    ax.barh([3], [np.mean(cshift_vals)], alpha=0.5)
    ax.scatter(cshift_vals, np.ones_like(cshift_vals)*3, color='k', s=10, alpha=0.5)
    ax.barh([2], [np.mean(platt_vals)], alpha=0.5)
    ax.scatter(platt_vals, np.ones_like(platt_vals)*2, color='k', s=10, alpha=0.5)
    ax.barh([1], [np.mean(cal_vals)], alpha=0.5)
    ax.scatter(cal_vals, np.ones_like(cal_vals)*1, color='k', s=10, alpha=0.5)
    ax.set_xlim(0, 0.3)
    ax.set_yticks([7, 6, 5, 4, 3, 2, 1])
    ax.set_yticklabels(['Train', 'CC', 'PCC(F1)', 'ACC', 'cshift', 'Platt', 'PCC(cal)'])
    """
    #filename = output_prefix + '_' + str(penalty) + '_' + str(max_n_train) + '.pdf'


def add_plot(ax, data, dataset, penalty, n_train, methods, method_names, xlabel):
    df = data[dataset]
    rows = df[(df.Penalty == penalty) & (df.n_train == n_train)]

    for i, method in enumerate(methods):
        vals = rows[rows.method == method].values[0, 4:]
        # filter extreme outliers
        #vals = [val for val in vals if val < 0.35]
        ax.barh([7-i], [np.mean(vals)], alpha=0.6)
        ax.scatter(vals, np.ones_like(vals)*(7-i), color='k', s=10, alpha=0.6)

    ax.set_xlim(0, 0.35)
    #ax.set_xscale('log', nonposx='clip')
    ax.set_yticks([7, 6, 5, 4, 3, 2, 1])
    ax.set_yticklabels(method_names)
    ax.set_xlabel(xlabel)


if __name__ == '__main__':
    main()


