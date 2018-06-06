import os
import re
import string
import sys
from collections import Counter
from optparse import OptionParser

import numpy as np
from scipy import sparse

from util import file_handling as fh

"""
Convert a dataset into the required format.
Input format is one line per item.
Each line should be a json object.
At a minimum, each json object should have a "text" field, with the document tex
If training and test data are to be processed separately, the same output directory should be used
and the --train_prefix option shoudl be used to specify the prefix used for the training data (for vocab, etc.)
Run "python preprocess_data -h" for more options.
If an 'id' field is provided, this will be used as an identifier in the dataframes, otherwise index will be used 
"""

# compile some regexes
# get basic punctuation excuding a few special characters
punct_chars = list(set(string.punctuation) - {"'", '.', '_'})
punct_chars.sort()
# add unicode dash characters
unicode_dashes = [u'\u2010', u'\u2011', u'\u2012', u'\u2013', u'\u2014', u'\u2015']
dashes = re.compile('[%s]' % re.escape(''.join(unicode_dashes + ['-'])))
punct_chars.extend(unicode_dashes)
punctuation = ''.join(punct_chars)
replace = re.compile('[%s]' % re.escape(punctuation))
alpha = re.compile('^[a-zA-Z_]+$')
alpha_or_num = re.compile('^[a-zA-Z_]+|[0-9_]+$')
alphanum = re.compile('^[a-zA-Z0-9_]+$')


def main():
    usage = "%prog data.jsonlist output_dir output_prefix"
    parser = OptionParser(usage=usage)
    parser.add_option('--strip_html', action="store_true", dest="strip_html", default=False,
                      help='Strip HTML tags: default=%default')
    parser.add_option('--bigrams', action="store_true", dest="bigrams", default=False,
                      help='Output bigrams instead of unigrams: default=%default')
    parser.add_option('--chargrams', dest='chargrams', default=0,
                      help='Use chargrams of order n instead of word tokens: default=%default')
    parser.add_option('--no_lower', action="store_true", dest="no_lower", default=False,
                      help='Do not lower case words: default=%default')
    parser.add_option('--replace_num', action="store_true", dest="replace_num", default=False,
                      help='Replace numbers with a __NUM__ symbol: default=%default')
    parser.add_option('--plain', action="store_true", dest="plain", default=False,
                      help='Output plaintext (one doc per line): default=%default')
    parser.add_option('--stopwords', dest='stopwords', default=None,
                      help='Stopwords [None|mallet|snowball]: default=%default')
    parser.add_option('--drop_quotes', action="store_true", dest="drop_quotes", default=False,
                      help='Remove quoted text: default=%default')
    parser.add_option('-d', dest='display', default=1000,
                      help='Display interval: default=%default')

    (options, args) = parser.parse_args()

    train_infile = args[0]
    output_dir = args[1]
    output_prefix = args[2]

    config = {}
    config['strip_html'] = options.strip_html
    config['bigrams'] = options.bigrams
    config['chargrams'] = int(options.chargrams)
    config['lower'] = not options.no_lower
    config['replace_num'] = options.replace_num
    config['plain'] = options.plain
    config['stopwords'] = options.stopwords
    config['display'] = int(options.display)
    config['remove_quotes'] = options.drop_quotes

    if not os.path.exists(output_dir):
        sys.exit("Error: output directory does not exist")

    preprocess_data_fast(train_infile, output_dir, output_prefix, config)


def preprocess_data_fast(infile, output_dir, output_prefix, config):
    strip_html = config['strip_html']
    lower = config['lower']
    output_plaintext = config['plain']
    bigrams = config['bigrams']
    replace_num = config['replace_num']
    chargrams = config['chargrams']
    remove_quotes = config['remove_quotes']
    display_interval = config['display']

    print("Reading data files")
    items = fh.read_jsonlist(infile)
    n_items = len(items)

    stopwords_name = config['stopwords']
    if stopwords_name is not None:
        stopwords = fh.read_text(os.path.join('stopwords', stopwords_name + '_stopwords.txt'))
        stopwords = set([word.strip() for word in stopwords])
    else:
        stopwords = set()

    print("Parsing %d documents" % n_items)
    vocab = set()
    ids = []
    token_counts = []
    word_strings = []
    quote_counts = []
    quote_chars = []
    n_tokens_list = []

    for i, item in enumerate(items):
        if i % display_interval == 0 and i > 0:
            print(i)

        if 'id' in item:
            ids.append(item['id'])
        else:
            ids.append(i)

        # get the text and convert to tokens
        text = item['text']
        if chargrams > 0:
            tokens = []
            for n in range(1, chargrams+1):
                tokens.extend([text[i:i+n] for i in range(len(text)-n+1)])
        else:
            tokens, q_count, q_chars = tokenize(text, strip_html, lower=lower, bigrams=bigrams, stopwords=stopwords, replace_num=replace_num, remove_quotes=remove_quotes)
            n_tokens = len(tokens)
            quote_counts.append(q_count)
            quote_chars.append(q_chars)
            n_tokens_list.append(n_tokens)
        vocab.update(tokens)

        # store the token counts
        counter = Counter()
        counter.update(tokens)
        token_counts.append(counter)

        # also store word strings if desired
        if output_plaintext:
            word_strings.append(' '.join(tokens))

    print("{:0.2f} tokens per document on average".format(np.mean(n_tokens_list)))

    vocab = list(vocab)
    vocab.sort()
    vocab_size = len(vocab)
    vocab_index = dict(zip(vocab, range(vocab_size)))
    print("Size of full vocabulary=%d" % vocab_size)

    # create an empty sparse matrix
    print("Converting to sparse matrix")
    counts = sparse.lil_matrix((n_items, vocab_size))
    for i, item in enumerate(items):
        index_count_pairs = {vocab_index[term]: count for term, count in token_counts[i].items() if term in vocab_index}
        if len(index_count_pairs) > 0:
            indices, item_counts = zip(*index_count_pairs.items())
            counts[i, indices] = item_counts
        else:
            print("skipping %s" % item)

    print("Total tokens = %d" % counts.sum())

    print("Saving data")
    if chargrams > 0:
        feature_name = 'c' + str(chargrams) + 'grams'
    elif bigrams:
        feature_name = 'bigrams'
    else:
        feature_name = 'unigrams'
    fh.write_to_json(vocab, os.path.join(output_dir, output_prefix + '.' + feature_name + '.vocab.json'))
    if bigrams:
        fh.save_sparse(counts, os.path.join(output_dir, output_prefix + '.' + feature_name + '.npz'))
    else:
        fh.save_sparse(counts, os.path.join(output_dir, output_prefix + '.' + feature_name + '.npz'))
    fh.write_to_json(ids, os.path.join(output_dir, output_prefix + '.' + feature_name + '.ids.json'))

    if output_plaintext:
        fh.write_list_to_text(word_strings, os.path.join(output_dir, output_prefix + '.plain.txt'))

    if remove_quotes:
        fh.write_to_json(quote_counts, os.path.join(output_dir, output_prefix + '.quote_counts.json'))
        fh.write_to_json(quote_chars, os.path.join(output_dir, output_prefix + '.quote_chars.json'))


def tokenize(text, strip_html=False, lower=True, bigrams=False, stopwords=None, replace_num=False, remove_quotes=False):

    text, quote_count, quote_chars = clean_text(text, strip_html, lower, remove_quotes=remove_quotes)
    tokens = text.split()

    if replace_num:
        tokens = ['__year__' if re.match('^19\d\d[\']*[s]*$', t) else t for t in tokens]
        tokens = ['__year__' if re.match('^20\d\d[\']*[s]*$', t) else t for t in tokens]
        tokens = ['__rank__' if re.match('^\d+st$', t) else t for t in tokens]
        tokens = ['__rank__' if re.match('^\d+nd$', t) else t for t in tokens]
        tokens = ['__rank__' if re.match('^\d+rd$', t) else t for t in tokens]
        tokens = ['__rank__' if re.match('^\d+th$', t) else t for t in tokens]
        tokens = ['__num__' if re.match('^[\d]+$', t) else t for t in tokens]
        tokens = ['__alphanum__' if re.match('^\S*\d\S*$', t) else t for t in tokens]

    if bigrams:
        tokens = [tokens[i] + '_' + tokens[i+1] for i in range(len(tokens)-1) if (tokens[i] not in stopwords and tokens[i+1] not in stopwords)]
    else:
        tokens = [t for t in tokens if t not in stopwords]

    return tokens, quote_count, quote_chars


def clean_text(text, strip_html=False, lower=True, emails='replace', at_mentions='replace', urls='replace', remove_quotes=False):
    # remove html tags
    if strip_html:
        text = re.sub(r'<[^>]+>', '', text)
    else:
        # replace angle brackets
        text = re.sub(r'<', '(', text)
        text = re.sub(r'>', ')', text)

    # pad with spaces
    text = ' ' + text + ' '

    # find and remove quotations
    quote_count = 0
    quote_chars = 0
    if remove_quotes:
        pairs = []
        # look for opening and closing double quotes
        quote_starts = [m.start() for m in re.finditer('\s"', text)]
        quote_ends = [m.start() for m in re.finditer('"\s', text)] + [len(text)]
        quote_count = len(quote_starts)

        # form these into pairs, including lines that have no closing quote (because they are carried over)
        for s_i, start in enumerate(quote_starts):
            if s_i < len(quote_starts)-1:
                next_start = quote_starts[s_i+1]
                quote_end_list = [e for e in quote_ends if e > start and e < next_start]
                if len(quote_end_list) == 0:
                    end = next_start-1
                else:
                    end = quote_end_list[0]
            else:
                quote_end_list = [e for e in quote_ends if e > start]
                end = quote_end_list[0]
            pairs.append((start, end+1))
        # excise the quotes, from end to beginning
        pairs.reverse()
        for pair in pairs:
            text = text[:pair[0]] + ' ' + text[pair[1]:]
            quote_chars += pair[1] - pair[0]

    # lower case
    if lower:
        text = text.lower()

    # eliminate email addresses
    if emails == 'replace':
        text = re.sub(r'\S+@\S+', ' __email__ ', text)
    elif emails == 'drop':
        text = re.sub(r'\S+@\S+', ' ', text)

    # eliminate @mentions
    if at_mentions == 'replace':
        text = re.sub(r'\s@\S+', ' __mention__ ', text)
    elif at_mentions == 'drop':
        text = re.sub(r'\s@\S+', ' ', text)

    # replace urls:
    if urls == 'replace':
        text = re.sub('https\S+\s', ' __url__ ', text)
    elif urls == 'drop':
        text = re.sub('https\S+\s', ' ', text)

    # remove commas
    text = re.sub(r',', '', text)

    # replace most puncutation with spaces
    text = replace.sub(' ', text)

    # replace double single quotes with spaces
    text = re.sub(r'\'\'', ' ', text)

    # replace elipses with spaces
    text = re.sub(r'\.\.\.', ' ', text)

    # open up parentheses
    # remove single quotes at the start and end of words
    text = re.sub(r'\s\'', ' ', text)
    text = re.sub(r'^\'', ' ', text)
    text = re.sub(r'\'\s', ' ', text)
    text = re.sub(r'\'$', ' ', text)

    # replace single interior periods with spaces (assuming they are supposed to be sentence breaks)
    # in detail: look for a space[any word chars].[any word chars]space and replace . with space
    text = re.sub(r'(\s\w+)\.(\w+\s)', r'\1 \2', text)

    # remove all other periods (hopefully preserving acronyms)
    text = re.sub(r'\.', '', text)

    # drop some lone puntuation
    text = re.sub(r'\s[_\']+\s', ' ', text)

    # replace all whitespace with a single space
    text = re.sub(r'\s', ' ', text)

    # strip off spaces on either end
    text = text.strip()

    return text, quote_count, quote_chars


if __name__ == '__main__':
    main()
