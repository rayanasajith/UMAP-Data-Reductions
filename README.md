UMAP Data Reduction Reproduction

Paper: UMAP: Uniform Manifold Approximation and Projection for Dimension Reduction (McInnes et al., 2020)

Data Preparation Context

In the context of data preparation, UMAP is a Data Reduction technique. High-dimensional datasets suffer from the "curse of dimensionality," making it difficult for machine learning algorithms to process them efficiently. UMAP prepares the data by reducing the ambient dimension while keeping the most important structural relationships intact.
To prove that the UMAP data preparation step was successful, the paper evaluates the embeddings by training a k-Nearest Neighbors classifier on the low-dimensional data. If the classifier achieves high accuracy, it demonstrates that UMAP successfully preserved the essential local structure of the dataset during the preparation phase.

Overview

This repository contains the code to reproduce the main quantitative experiment from the UMAP paper (Table 1 and Section 5.2). The reproduction was performed on two datasets specified in the paper:

1. PenDigits: 1,797 grayscale images (8x8), flattened to 64 dimensions.
2. COIL-20: 1,440 grayscale images (128x128), flattened to 16,384 dimensions.

Prerequisites

1. Python 3.10+
2. The required packages are listed in requirements.txt. The main dependencies are umap-learn, scikit-learn, numpy, imageio, and requests.

Setup Instructions

1. Create a virtual environment.
2. Activate the virtual environment.
3. Install the required packages using the provided requirements.txt

How to Run the Experiments

To reproduce the results, run the two scripts separately:
1. Run the PenDigit.py script.
2. Run the coil20.py script.

Parameters Used

The reproduction uses the default parameters recommended in the paper:

n_neighbors = 15
min_dist = 0.1
n_components = 2
random_state = 42 (added for reproducibility)
k (kNN classifier) = 10, 20, 40, 80, 160
Cross-validation = 10-fold Stratified

Expected Output

Both scripts will print a table containing the average accuracy and standard deviation for each value of k. These results can be directly compared to Table 1 in the original paper.
