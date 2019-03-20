# Slim

This is a compression algorithm that iteratively builds a code table of frequent itemsets.  This is a scikit-learn-compatible, partial implementation of Smets and Vreeken (2012), SLIM: Directly mining description patterns, SDM12.

# SlimClassifier

This creates a Slim transformer for each class, then predicts by finding the transformer that achieves best compression.