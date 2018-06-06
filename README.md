This is a repo to accompany the paper The Importance of Calibration for Estimating Proportions from Annotations. It implements a few different methods for quantification, but is provided primarily for the purpose of replicabilty, and is not intended to be used in production.

### Requirements

- python3
- numpy
- scipy
- pandas
- scikit-learn
- matplotlib
- seaborn

### Datasets

- [Media Frames Corpus](http://www.cs.cmu.edu/~dcard/resources/card.acl2015.pdf)
- [Amazon review datasets](http://jmcauley.ucsd.edu/data/amazon/)
- Yelp10 [no longer available from Yelp]
- [Sentiment140](http://help.sentiment140.com/for-students/)

### Preprocessing

For each dataset, there is an import script to convert documents to json objects. For example, run:

`python -m datasets.import_amazon /m-pinotHD/dallas/data/amazon_reviews/reviews_Clothing_Shoes_and_Jewelry_5.json.gz data/amazon/clothes/`

Then call prepocess_data_fast twice to extract unigrams and bigrams:

`python -m preprocessing.preprocess_data_fast data/amazon/clothes/all.jsonlist data/amazon/clothes all`
`python -m preprocessing.preprocess_data_fast data/amazon/clothes/all.jsonlist data/amazon/clothes all --bigrams`

Any other features could also be used, if they are converted to the same format.

### Experiments

The `run_lr_grid.py` method can be used to run a variety of experiments, and will use all methods in the paper to estimate proportions. To re-create experiments, use the scripts in the `experiments` directory to generate scripts to run many experiments, and then use `post/average_over_dev_folds.py` to combine the predicted proportions on the test set from all folds of train/dev cross-validation.


### References

Dallas Card and Noah A. Smith. The Importance of Calibration for Estimating Proportions from Annotations. In Proceedings of NAACL, 2018. [paper] [supplementary] [BibTeX]