import os
import sys
from collections import Counter
from optparse import OptionParser

import numpy as np
import pandas as pd

from util import file_handling as fh

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

CODES = {str(int(i+1)): f for i, f in enumerate(FRAMES)}

n_frames = len(FRAMES)


def main():
    usage = "%prog path/to/documents.json metadata.json output_dir"
    parser = OptionParser(usage=usage)
    #parser.add_option('-y', dest='year', default=2004,
    #                  help='Year at which to divide data: default=%default')
    #parser.add_option('--boolarg', action="store_true", dest="boolarg", default=False,
    #                  help='Keyword argument: default=%default')

    (options, args) = parser.parse_args()

    data_file = args[0]
    metadata_file = args[1]
    output_dir = args[2]

    if not os.path.exists(output_dir):
        sys.exit("Output directory does not exist")

    data = fh.read_json(data_file)
    metadata = fh.read_json(metadata_file)

    total_frames = np.zeros([n_frames, 2])

    keys = list(data.keys())
    keys.sort()

    year_counts = Counter()

    text_lines = []
    year_list = []
    annotations = {}
    for frame in FRAMES:
        annotations[frame] = []

    numbers_of_annotations = []

    count = 0
    for k in keys:
        irrelevant = data[k]['irrelevant']
        if irrelevant is not None and not int(irrelevant):
            n_annotations = 0
            text = data[k]['text']
            paragraphs = text.split('\n\n')
            text = ' '.join(paragraphs[2:])
            framing_annotations = data[k]['annotations']['framing']
            frame_counts = np.zeros([n_frames, 2])

            # process framing annotations
            for annotator, annotation_list in framing_annotations.items():
                annotator_frames = np.zeros(n_frames)
                # look for presence of each frame
                for a in annotation_list:
                    frame = int(a['code']) - 1
                    annotator_frames[frame] = 1

                # note the presence or absence of each frame for this annotator
                frame_counts[:, 1] += annotator_frames
                frame_counts[:, 0] += 1 - annotator_frames

            year = int(metadata[k]['year'])

            if np.sum(frame_counts) > 0 and year >= 1990:
                total_frames += frame_counts

                # keep all annotations
                text_lines.append({'text': text, 'id': count, 'orig_id': k, 'year': int(year)})

                for frame_i, frame in enumerate(FRAMES):
                    annotations[frame].append([frame_counts[frame_i, 0], frame_counts[frame_i, 1]])
                count += 1

                year_list.append(year)
                year_counts.update([year])

                numbers_of_annotations.append(np.sum(frame_counts[1, :]))

    n_articles = len(text_lines)
    print("Found {:d} useable articles".format(n_articles))

    print("Distribution of annotations")
    annotation_dist = np.bincount(numbers_of_annotations)
    print(annotation_dist)

    fh.write_jsonlist(text_lines, os.path.join(output_dir, 'all.jsonlist'))

    for frame_i, frame in enumerate(FRAMES):
        print(frame, total_frames[frame_i], total_frames[frame_i, :].sum(), total_frames[frame_i, 1] / total_frames[frame_i, :].sum())
        df = pd.DataFrame(np.vstack(annotations[frame]), index=np.arange(n_articles), columns=['0', '1'])
        df.to_csv(os.path.join(output_dir, 'all.' + frame + '.csv'))

    df = pd.DataFrame(year_list, index=np.arange(n_articles), columns=['year'])
    df.to_csv(os.path.join(output_dir, 'all.years.csv'))

    print("Year counts:")
    keys = list(year_counts.keys())
    keys.sort()
    count = 0
    for k in keys:
        count += year_counts[k]
        print(k, year_counts[k], count)


if __name__ == '__main__':
    main()

