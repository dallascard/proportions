This is a repo to accompany the paper [The Importance of Calibration for Estimating Proportions from Annotations](https://www.cs.cmu.edu/~dcard/resources/NAACL_2018_main.pdf). It implements a few different methods for quantification, but is provided primarily for the purpose of replicabilty, and is not intended to be used in production.

### Requirements

- python3
- numpy
- scipy
- pandas
- scikit-learn
- matplotlib
- seaborn

### Datasets

- [Media Frames Corpus](http://www.cs.cmu.edu/~dcard/resources/card.acl2015.pdf) [contact me]
- [Amazon review datasets](http://jmcauley.ucsd.edu/data/amazon/) [contact Julian McAuley]
- Yelp10 [no longer available from Yelp]
- [Sentiment140](http://help.sentiment140.com/for-students/)

### Preprocessing

For each dataset, there is an import script to convert documents to json objects. For example, run:

`python -m datasets.import_amazon downloads/amazon_reviews/reviews_Clothing_Shoes_and_Jewelry_5.json.gz data/amazon/clothes/`

Then call prepocess_data_fast twice to extract unigrams and bigrams:

```
python -m preprocessing.preprocess_data_fast data/amazon/clothes/all.jsonlist data/amazon/clothes all
python -m preprocessing.preprocess_data_fast data/amazon/clothes/all.jsonlist data/amazon/clothes all --bigrams
```

Any other features could also be used, if they are converted to the same format.

### Experiments

The `run_lr_grid.py` method can be used to run a variety of experiments. For example, to run one dev-fold of one sub experiment of the Amazon data, with l1 regularization, 500 training documents, and calibration as a criterion for model selection, use:

`python run_lr_grid.py data/amazon/clothing all  --label labels --features unigrams,bigrams --min_dfs 0,0 --penalty l1 --objective calibration --max_n_train 500 --metadata metadata --field year --train_start 2010 --train_end 2010 --test_start 2011 --test_end 2011 --dev_fold 0`

To re-create experiments, use the scripts in the `experiments` directory to generate scripts to run all the experiments, run all the resulting bash scripts, and then use `post/average_over_dev_folds.py` to combine the predicted proportions on the test set from all folds of train/dev cross-validation.


### References

Dallas Card and Noah A. Smith. The Importance of Calibration for Estimating Proportions from Annotations. In Proceedings of NAACL, 2018. [[paper](https://www.cs.cmu.edu/~dcard/resources/NAACL_2018_main.pdf)] [[supplementary](https://www.cs.cmu.edu/~dcard/resources/NAACL_2018_supplementary.pdf)] [[BibTeX](https://github.com/dallascard/proportions/blob/master/proportions.bib)]