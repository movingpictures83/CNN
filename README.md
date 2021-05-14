# CNN
# Language: Python
# Input: TXT
# Output: PREFIX
# Tested with: PluMA 1.1, Python 3.6
# DependenciesL tensorflow_1.14.0, numpy_1.16.0

Run a Convolutional Neural Network (CNNLawrence et al 1996) on a set of images.  Plugin assumes the images are broken into a training and testset.  It generates output classification probabilities for each image in the test set, its most likely classification, and prints accuracies to the screen.

The plugin expects as input a TXT file of tab-delimited keyword-value pairs:

classnames: File of possible classifications (TXT)
trainset: List of training images (CSV)
testset: List of test images (CSV)
tensor: CNN Tensors (one per line, TSV)
dense: Dense Layers (one per line, TSV)
optimize: Optimization strategy
metric: Metric to apply
epochs: Number of epochs

Output probabilities and final classifications will be generated as CSV files:

PREFIX.probs.csv: CSV file of classification probabilities (images are rows, classification groups are columns)
PREFIX.final.csv: Official classification of each image
