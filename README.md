# Slim

This is a compression algorithm that iteratively builds a code table of frequent itemsets.  This is a scikit-learn-compatible, implementation of Smets and Vreeken (2012), SLIM: Directly mining description patterns, SDM12.

Slim is especially useful in creating compression-based classifiers and outlier detectors on categorical and transactional data.  Continuous features are best supported with binning.  The unit tests use KBinsDiscretizer for this purpose.  This requires scikit-learn >= 0.20 to run the tests.

# SlimClassifier

This creates a Slim transformer for each class, then predicts by finding the transformer that achieves best compression.

# SlimOutlierDetector

This creates a Slim transformer for the positive class.  Negative examples are those which do not compress well.  The threshold for determining class is automatically determined using Cantelli's Inequality.