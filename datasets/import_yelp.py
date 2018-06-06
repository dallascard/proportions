import os
import re
import sys
from collections import Counter
from optparse import OptionParser

import numpy as np
import pandas as pd

from util import file_handling as fh


def main():
    usage = "%prog input_dir output_dir"
    parser = OptionParser(usage=usage)
    #parser.add_option('--keyword', dest='key', default=None,
    #                  help='Keyword argument: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    input_dir = args[0]
    output_dir = args[1]

    if not os.path.exists(output_dir):
        sys.exit("Error: Output directory does not exist")

    city_lookup = dict()

    print("Reading in business data")
    lines = fh.read_jsonlist(os.path.join(input_dir, 'business.json'))
    for line in lines:
        city = line['city']
        business_id = line['business_id']
        city_lookup[business_id] = city

    city_counts = Counter()
    print("Reading in review data")
    lines = fh.read_jsonlist(os.path.join(input_dir, 'review.json'))

    pairs = [('Las Vegas', 'Phoenix'), ('Toronto', 'Scottsdale'), ('Charlotte', 'Pittsburgh'), ('Tempe', 'Henderson')]

    for pair in pairs:
        text_lines = []
        labels = []
        years = []
        year_counts = Counter()
        count = 0
        city1, city2 = pair
        for i, line in enumerate(lines):
            if i % 100000 == 0:
                print(i, count)
            review_id = line['review_id']
            text = line['text']
            date = line['date']
            year = date.split('-')[0]
            funny = int(line['funny'])
            useful = int(line['useful'])
            cool = int(line['cool'])
            business_id = line['business_id']
            if business_id in city_lookup:
                city = city_lookup[business_id]
                city_counts.update([city])
                label = None
                if city == city1:
                    label = [1, 0]
                elif city == city2:
                    label = [0, 1]
                if label is not None:
                    text_lines.append({'text': text, 'city': city, 'year': year, 'id': count, 'review_id': review_id, 'label': label, 'funny': funny, 'useful': useful, 'cool': cool})
                    labels.append(label)
                    years.append(year)
                    year_counts.update([year])
                    count += 1

        n_reviews = len(text_lines)
        print(pair)
        print("Found {:d} reviews".format(n_reviews))
        name = '_'.join([re.sub('\s', '_', city) for city in pair])
        fh.write_jsonlist(text_lines, os.path.join(output_dir, name + '.jsonlist'))

        labels_df = pd.DataFrame(np.vstack(labels), index=np.arange(n_reviews), columns=[city1, city2])
        labels_df.to_csv(os.path.join(output_dir, name + '.labels.csv'))

        years_df = pd.DataFrame(years, index=np.arange(n_reviews), columns=['year'])
        years_df.to_csv(os.path.join(output_dir, name + '.years.csv'))

        print("Year counts")
        keys = list(year_counts.keys())
        keys.sort()
        for k in keys:
            print(k, year_counts[k])


if __name__ == '__main__':
    main()
