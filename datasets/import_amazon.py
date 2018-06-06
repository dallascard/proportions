import os
from collections import Counter
from optparse import OptionParser

import numpy as np
import pandas as pd

from util import file_handling as fh


def main():
    usage = "%prog reviews_file.json[.gz] output_dir"
    parser = OptionParser(usage=usage)
    parser.add_option('-p', dest='output_prefix', default='all',
                      help='prefix for output files: default=%default')
    parser.add_option('-n', dest='n_to_keep', default=None,
                      help='Only keep this many instances (None = all): default=%default')
    parser.add_option('-d', dest='display', default=1000,
                      help='Display progress after each batch of this many items: default=%default')

    (options, args) = parser.parse_args()

    reviews_file = args[0]
    output_dir = args[1]
    output_prefix = options.output_prefix
    n_to_keep = options.n_to_keep
    dislay = int(options.display)

    import_review_data(reviews_file, output_dir, output_prefix, n_to_keep, dislay)


def import_review_data(reviews_file, output_dir, output_prefix, n_to_keep, display=1000):
    print("Loading data")
    reviews = fh.read_jsonlist(reviews_file)

    n_items = len(reviews)
    if n_to_keep is not None:
        n_to_keep = int(n_to_keep)
    else:
        n_to_keep = n_items

    print("Loaded %d items" % n_items)

    dates = pd.DataFrame(columns=['date'])
    reviewers = set()
    asins = set()
    year_counts = Counter()

    #if prop < 1.0:
    #    subset_size = int(prop * n_items)
    #    subset = np.random.choice(range(n_items), size=subset_size, replace=False)
    #    keys = [keys[i] for i in subset]
    #    print("Using a random subset of %d reviews" % subset_size)

    votes = []
    years = []
    lines = []
    numbers_of_votes = []

    count = 0

    for k_i, review in enumerate(reviews):
        if k_i % display == 0:
            print(k_i)
        helpfulness = review['helpful']
        n_helpful_votes = helpfulness[0]
        n_votes = helpfulness[1]
        # skip items with no votes, and errors where n_helpful > n_votes
        if n_votes > 0 and n_helpful_votes <= n_votes:
            numbers_of_votes.append(n_votes)
            date_string = review['reviewTime']
            parts = date_string.split(',')
            year = int(parts[1])
            parts2 = parts[0].split()
            month = int(parts2[0])
            day = int(parts2[1])
            if year > 2006:
                data = {}
                asins.add(review['asin'])
                reviewers.add(review['reviewerID'])
                data['id'] = count
                data['orig_line'] = k_i
                data['text'] = review['summary'] + '\n\n' + review['reviewText']
                data['rating'] = review['overall']
                data['helpfulness'] = [n_votes - n_helpful_votes,  n_helpful_votes]
                year_counts.update([year])
                data['year'] = year
                years.append(year)
                votes.append([n_votes-n_helpful_votes, n_helpful_votes])
                lines.append(data)
                date = pd.Timestamp(year=year, month=month, day=day)
                dates.loc[k_i] = date
                count += 1
        if count >= n_to_keep:
            break

    n_reviews = len(lines)
    print("Found %d useable reviews" % n_reviews)

    print("Earliest date:", dates.date.min())
    print("Latest date:", dates.date.max())
    print("%d reviewers" % len(reviewers))
    print("%d products" % len(asins))

    vote_dist = np.bincount(numbers_of_votes)
    print(vote_dist[:10], np.sum(vote_dist[10:]))

    print("Year counts:")
    keys = list(year_counts.keys())
    keys.sort()
    for k in keys:
        print(k, year_counts[k])

    print("Saving data")
    fh.makedirs(output_dir)
    fh.write_jsonlist(lines, os.path.join(output_dir, output_prefix + '.jsonlist'))

    metadata_df = pd.DataFrame(years, index=np.arange(n_reviews), columns=['year'])
    labels_df = pd.DataFrame(votes, index=np.arange(n_reviews), columns=['not_helpful', 'helpful'])

    metadata_df.to_csv(os.path.join(output_dir, output_prefix + '.metadata.csv'))
    labels_df.to_csv(os.path.join(output_dir, output_prefix + '.labels.csv'))

    fh.write_to_json(reviews[0], os.path.join(output_dir, 'article0.json'))

if __name__ == '__main__':
    main()
