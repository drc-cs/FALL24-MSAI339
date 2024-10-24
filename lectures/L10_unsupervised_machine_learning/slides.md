---
title: MSAI 339
separator: <!--s-->
verticalSeparator: <!--v-->
theme: serif
revealOptions:
  transition: 'none'
---

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 70%; position: absolute;">

  # Data Science
  ## L.10 | Unsupervised Machine Learning

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 80%; padding-top: 30%">
  <img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*WunYbbKjzdXvw73a4Hd2ig.gif" height="100%" style="margin: 0 auto; display: block;">
  <p style="text-align: center; font-size: 0.6em; color: grey;">Seo, 2023</p>
  </div>
</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 70%; position: absolute;">

  # Welcome to MSAI 339.
  ## Please check in using PollEverywhere.
  Scan the QR code or go to [pollev.com/nucs](https://pollev.com/nucs)

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->

## Announcements

- H.03 is due tonight at 11:59 PM.

- Spread out homework pacing.
  - H.04 will now be due 11.12, and H.05 will now be due 11.21.

- Final 'quiz'.

<!--s-->

## XGBoost Follow-Up | Recap

**Gradient Boosting** is another popular boosting algorithm. Gradient Boosting works by fitting a sequence of weak learners to the residuals of the previous model. This differs from AdaBoost, which focuses on the misclassified instances. A popular implementation of Gradient Boosting is **XGBoost** (❤️).

**Key Features of Gradient Boosting**:

1. Fit a weak learner to the training data.
2. Compute the residuals of the model.
3. Fit the next weak learner to the residuals.
4. Repeat the process until a stopping criterion is met.

<img src="https://media.geeksforgeeks.org/wp-content/uploads/20200721214745/gradientboosting.PNG" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Geeksforgeeks</p>

<!--s-->

## XGBoost Follow-Up | Questions

1. Why do they call it **gradient** boosting?
    - With XGBoost, we compute the gradient of the loss function with respect to the model parameters.
    - XGBoost uses the gradient to update the model in the direction that minimizes the loss function.

2. What is the **learning rate** doing? 
    - The learning rate controls the step size when updating the model.
    - A smaller learning rate results in a more conservative update, while a larger learning rate results in a more aggressive update.
    - In terms of residuals, a smaller learning rate will reduce the impact of each weak learner on the final model.

<!--s-->

## XGBoost Follow-Up | Original Paper

XGBoost was introduced by Tianqi Chen and Carlos Guestrin in 2016. The paper is titled "XGBoost: A Scalable Tree Boosting System" and is available [here](https://arxiv.org/pdf/1603.02754).

<img src="https://storage.googleapis.com/slide_assets/xgboost.png" width="70%" style="margin: 0 auto; display: block; border-radius: 10px;">

<!--s-->

## XGBoost Follow-Up | Scratch Implementation

<div style = "font-size: 0.8em;">

I followed the [original paper](https://arxiv.org/pdf/1603.02754) and reformatted structure from this [implementation](https://randomrealizations.com/posts/xgboost-from-scratch/) for clarity. Equations taken directly from the paper are referenced in the code, and performance is ~ identical to the python package <span class="code-span">xgboost</span>. 

I hope this helps illuminate how XGBoost (❤️) works for the curious!

</div>

<div style = "width: 100%;">


```python

"""
This script contains an implementation of the XGBoost algorithm.
The original XGBoost paper by Chen and Guestrin (2016): https://arxiv.org/pdf/1603.02754
Numpy code borrows heavily from https://randomrealizations.com/posts/xgboost-from-scratch/.
The code is written for educational purposes and is not optimized for performance.

Joshua D'Arcy, 2024.
"""

import math
import numpy as np 
from dataclasses import dataclass
from typing import Callable

@dataclass
class Parameters:
    learning_rate: float = 0.1
    max_depth: int = 5
    subsample: float = 0.8
    reg_lambda: float = 1.5
    gamma: float = 0.0
    min_child_weight: float = 25
    base_score: float = 0.0
    random_seed: int = 42

class XGBoost():
    
    def __init__(self, parameters: Parameters):
        self.params = parameters
        self.base_prediction = self.params.base_score if self.params.base_score else 0.5
        self.rng = np.random.default_rng(seed=parameters.random_seed)
                
    def fit(self, X: np.ndarray, y: np.ndarray, objective: Callable, num_boost_round: int):

        # Initialize the base prediction.
        current_predictions = self.base_prediction * np.ones(shape=y.shape)

        # Initialize the list of boosters.
        self.boosters = []

        # Iterate over the number of boosting rounds.
        for i in range(num_boost_round):

            # Compute the first and second order gradients.
            first_order = objective.first_order(y, current_predictions)
            second_order = objective.second_order(y, current_predictions)

            # Get the sample indices (if subsampling is used).
            if self.params.subsample == 1.0:
                sample_idxs = np.arange(len(y))
            else:
                sample_idxs = self.rng.choice(len(y), size=math.floor(self.params.subsample*len(y)), replace=False)

            # Train a new Tree (booster) on the gradients and hessians.
            booster = Tree(X, first_order, second_order, self.params, sample_idxs)

            # Update the current predictions.
            current_predictions += self.params.learning_rate * booster.predict(X)

            # Append the new booster to the list of boosters.
            self.boosters.append(booster)
            
    def predict(self, X: np.ndarray):
        
        # Return the final prediction, based on the base prediction and the sum of the predictions of all boosters.
        summed_predictions = np.sum([booster.predict(X) for booster in self.boosters], axis=0)
        return (self.base_prediction + self.params.learning_rate * summed_predictions)

class Tree:
    def __init__(self, X: np.ndarray, first_order: np.ndarray, second_order: np.ndarray, parameters: Parameters, indices: np.ndarray, current_depth: int = 0):
        # Initialize the parameters.
        self.params = parameters
        self.X = X
        self.first_order = first_order
        self.second_order = second_order
        self.idxs = indices
        self.n = len(indices)
        self.c = X.shape[1]
        self.current_depth = current_depth

        # Equation (5) in the XGBoost paper.
        self.value = -first_order[indices].sum() / (second_order[indices].sum() + self.params.reg_lambda)

        # Set the initial best score to 0.
        self.best_score_so_far = 0.0

        # Initialize the left and right child nodes if max depth is not reached.
        if self.current_depth < self.params.max_depth:
            self.insert_child_nodes()

    def insert_child_nodes(self):
        # Find the best split for each feature.
        for i in range(self.c):
            self.find_split(i)

        # Check if the current node is a leaf node.
        if self.best_score_so_far == 0.0:
            return

        # Get x, the feature values for the current node.
        x = self.X[self.idxs, self.split_feature_idx]

        # Get the indices of the left and right child nodes.
        left_idx = np.nonzero(x <= self.threshold)[0]
        right_idx = np.nonzero(x > self.threshold)[0]

        # Create the left and right child nodes, incrementing depth.
        self.left = Tree(self.X, self.first_order, self.second_order, self.params, self.idxs[left_idx], self.current_depth + 1)
        self.right = Tree(self.X, self.first_order, self.second_order, self.params, self.idxs[right_idx], self.current_depth + 1)

    def find_split(self, feature_idx: int):

        # Get the feature values for the current node.
        x = self.X[self.idxs, feature_idx]

        # Sort the feature values.
        first = self.first_order[self.idxs]
        second = self.second_order[self.idxs]
        sort_idx = np.argsort(x)
        sort_first = first[sort_idx]
        sort_second = second[sort_idx]
        sort_x = x[sort_idx]

        # Initialize the sum of the first and second order gradients.
        sum_first = first.sum()
        sum_second = second.sum()
        sum_first_right = sum_first
        sum_second_right = sum_second
        sum_first_left = 0.0
        sum_second_left = 0.0

        for i in range(0, self.n - 1):

            # Get the first and second order gradients for the current split.
            first_i = sort_first[i]
            second_i = sort_second[i]
            x_i = sort_x[i]
            x_i_next = sort_x[i + 1]

            # Update the sum of the first and second order gradients.
            sum_first_left += first_i
            sum_first_right -= first_i
            sum_second_left += second_i
            sum_second_right -= second_i

            # Skip if the current split does not meet the minimum child weight requirement.
            if sum_second_left < self.params.min_child_weight or x_i == x_i_next:
                continue

            if sum_second_right < self.params.min_child_weight:
                break

            # Compute the gain of the current split, Equation (7) in the XGBoost paper.
            first_term = sum_first_left ** 2 / (sum_second_left + self.params.reg_lambda)
            second_term = sum_first_right ** 2 / (sum_second_right + self.params.reg_lambda)
            third_term = sum_first ** 2 / (sum_second + self.params.reg_lambda)
            gain = 0.5 * (first_term + second_term - third_term) - self.params.gamma / 2

            # Update the best split if the current split is better.
            if gain > self.best_score_so_far:
                self.split_feature_idx = feature_idx
                self.best_score_so_far = gain
                self.threshold = (x_i + x_i_next) / 2

    def predict(self, X: np.ndarray):
        # Iterate over each row in the input data and make a prediction.
        return np.array([self.predict_row(row) for row in X])

    def predict_row(self, row: np.ndarray):
        # Check if the current node is a leaf node.
        if self.best_score_so_far == 0.0:
            return self.value

        # If the current node is not a leaf node, then we need to find the child node.
        child = self.left if row[self.split_feature_idx] <= self.threshold else self.right

        # Recursively call the predict_row method on the child node.
        return child.predict_row(row)

```

</div>


<!--s-->



<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you working with **clustering** algorithms?

  Scan the QR code or go to [pollev.com/nucs](https://pollev.com/nucs)

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->

<div class="header-slide">

# L.10 | Unsupervised Machine Learning

</div>

<!--s-->

<div style = "font-size: 0.8em;">

# Industry Application

### Scenario

You are a data scientist at a large e-commerce company. The company has a large dataset of customer transactions and wants to segment customers based on their purchase history. The goal is to identify groups of customers with similar purchasing behavior, so that the company can tailor marketing strategies to each segment.

### Problem Statement

Segment customers based on their purchase history using unsupervised machine learning techniques.

### I/O

- **Input**: Customer transaction data (e.g., purchase amount, purchase frequency).
- **Output**: Customer segments based on purchasing behavior.

### Goal

Identify groups of customers with similar purchasing behavior using unsupervised machine learning techniques.

### Plan

Where do you start?

</div>

<!--s-->

## Agenda

Supervised machine learning is about predicting $y$ given $X$. Unsupervised machine learning is about finding patterns in $X$, without being given $y$.

There are a number of techniques that we can consider to be unsupervised machine learning, spanning a broad range of methods (e.g. clustering, dimensionality reduction, anomaly detection). In this lecture, we will focus on clustering.

### Clustering
1. Partitional Clustering
2. Hierarchical Clustering
3. Density-Based Clustering

<!--s-->

<div class="header-slide">

# Clustering

</div>

<!--s-->

## Clustering | Applications

Clustering is a fundamental technique in unsupervised machine learning, and has a wide range of applications.

<div class = "col-wrapper">
<div class="c1" style = "width: 70%; font-size: 0.75em;">

<div>

| Application | Example | 
| --- | --- |
| Customer segmentation | Segmenting customers based on purchase history. |
| Document clustering | Grouping similar documents together. |
| Image segmentation | Segmenting an image into regions of interest. |
| Anomaly detection | Detecting fraudulent transactions. |
| Recommendation systems | Recommending products based on user behavior. |
</div>

</div>
<div class="c2" style = "width: 30%">

<div>
<img src="https://cambridge-intelligence.com/wp-content/uploads/2021/01/graph-clustering-800px.png" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Cambridge Intelligence, 2016</p>
</div>

</div>
</div>

<!--s-->

## Clustering | Goals

Clustering is the task of grouping a set of objects in such a way that objects in the same group (or cluster) are more similar to each other than to those in other groups.

A good clustering result has the following properties:

- High intra-cluster similarity
- Low inter-cluster similarity

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*vEsw12wO0KxvYne0m4Cr5w.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Yehoshua, 2023</p>

<!--s-->

## Clustering | Revisiting Distance Metrics

> "Everything in data science seems to come down to distances and separating stuff." - Anonymous PhD

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; font-size: 0.75em;">

### Euclidean Distance

`$ d(x, x') =\sqrt{\sum_{i=1}^n (x_i - x'_i)^2} $`

### Manhattan Distance

`$ d(x, x') = \sum_{i=1}^n |x_i - x'_i| $`

### Cosine Distance 

`$ d(x, x') = 1 - \frac{x \cdot x'}{||x|| \cdot ||x'||} $`

</div>

<div class="c2" style = "width: 50%; font-size: 0.75em;">

### Jaccard Distance (useful for categorical data!)

`$ d(x, x') = 1 - \frac{|x \cap x'|}{|x \cup x'|} $`

### Hamming Distance (useful for strings!)

`$ d(x, x') = \frac{1}{n} \sum_{i=1}^n x_i \neq x'_i $`

</div>
</div>

<!--s-->

## Clustering | Properties

<div style="font-size: 0.75em;">

To determine the distance between two points (required for clustering), we need to define a distance metric. A good distance metric has the following properties:

1. Non-negativity: $d(x, y) \geq 0$
    - Otherwise, the distance between two points could be negative.
    - If Bob is 5 years old and Alice is 10 years old, the distance between them could be -5 years.

2. Identity: $d(x, y) = 0$ if and only if $x = y$
    - Otherwise, the distance between two points could be zero even if they are different.
    - If Bob is 5 years old and Alice is 5 years old, the distance between them should be zero.

3. Symmetry: $d(x, y) = d(y, x)$
    - Otherwise, the distance between two points could be different depending on the order.
    - If the distance between Bob and Alice is 5 years, the distance between Alice and Bob should also be 5 years.

4. Triangle inequality: $d(x, y) + d(y, z) \geq d(x, z)$
    - Otherwise, the distance between two points could be shorter than the sum of the distances between intermediate points.
    - If the distance between Bob and Alice is 5 years and the distance between Alice and Charlie is 5 years, then the distance between Bob and Charlie should be at most 10 years. We don’t want shortcuts in our triangles.

</div>

<!--s-->

## Clustering Approaches

Okay, so we have a good understanding of distances. Now, let's talk about the different approaches to clustering data without labels using those distances. There are **many** clustering algorithms, but they can be broadly categorized into a few main approaches:


<img src="https://www.mdpi.com/sensors/sensors-23-06119/article_deploy/html/images/sensors-23-06119-g003.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Adnan, 2023</p>

<!--s-->

## Clustering Approaches

### Partitional Clustering
- Partitional clustering divides a dataset into $k$ clusters, where $k$ is a tunable hyperparameter.
- Example algorithm: K-Means

### Hierarchical Clustering
- Hierarchical clustering builds a hierarchy of clusters, which can be visualized as a dendrogram.
- Example algorithm: Agglomerative Clustering

### Density-Based Clustering
- Density-based clustering groups together points that are closely packed in the feature space.
- Example algorithm: DBSCAN

<!--s-->

<div class="header-slide">

# Partitional Clustering

</div>

<!--s-->

## Partitional Clustering | Introduction

Partitional clustering is the task of dividing a dataset into $k$ clusters, where $k$ is a hyperparameter. The goal is to minimize the intra-cluster distance and maximize the inter-cluster distance, subject to the constraint that each data point belongs to exactly one cluster.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*vEsw12wO0KxvYne0m4Cr5w.png" height="50%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Yehoshua, 2023</p>

<!--s-->

## Partitional Clustering | K-Means

K-Means is a popular partitional clustering algorithm that partitions a dataset into $k$ clusters. The algorithm works as follows:

```text
1. Initialize $k$ cluster centroids randomly.
2. Assign each data point to the nearest cluster centroid.
3. Update the cluster centroids by taking the mean of the data points assigned to each cluster.
4. Repeat steps 2 and 3 until convergence.
    - Convergence occurs when the cluster centroids do not change significantly between iterations.
```


<img src="https://ben-tanen.com/assets/img/posts/kmeans-cluster.gif" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Tanen, 2016</p>

<!--s-->

## Partitional Clustering | K-Means

K-Means is a simple and efficient algorithm, but it has some limitations:

- The number of clusters $k$ must be specified in advance.
- The algorithm is sensitive to the initial cluster centroids.
- The algorithm may converge to a local minimum.

There are variations of K-Means that help address limitations (e.g., K-Means++, MiniBatch K-Means).

Specifying the number of clusters $k$ is a common challenge in clustering, but there are techniques to estimate an optimal $k$, such as the **elbow method** and the **silhouette score**. Neither of these techniques is perfect, but they can be informative under the right conditions.

<!--s-->

## Partitional Clustering | K-Means Elbow Method

The elbow method is a technique for estimating the number of clusters $k$ in a dataset. The idea is to plot the sum of squared distances between data points and their cluster centroids as a function of $k$, and look for an "elbow" in the plot.

```text
1. Run K-Means for different values of $k$.
2. For each value of $k$, calculate the sum of squared distances between data points and their cluster centroids.
3. Plot the sum of squared distances as a function of $k$.
4. Look for an "elbow" in the plot, where the rate of decrease in the sum of squared distances slows down.
5. The number of clusters $k$ at the elbow is a good estimate.
```

<img src="https://media.licdn.com/dms/image/D4D12AQF-yYtbzPvNFg/article-cover_image-shrink_600_2000/0/1682277078758?e=2147483647&v=beta&t=VhzheKDjy7bEcsYyrjql3NQAUcTaMBCTzhZWSVVSeNg" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Etemadi, 2020</p>

<!--s-->

## Partitional Clustering | K-Means Silhouette Score

The silhouette score measures how similar an object is to its own cluster (cohesion) compared to other clusters (separation). The optimal silhouette score is 1, and the worst score is -1. When using the silhouette score to estimate the number of clusters $k$, we look for the value of $k$ that maximizes the silhouette score.

$$ s = \frac{b - a}{\max(a, b)} $$

Where: 
- $a$ is the mean distance between a sample and all other points in the same cluster.
- $b$ is the mean distance between a sample and all other points in the next nearest cluster.

Pseudocode for estimating $k$ using the silhouette score:
```text
1. Run K-Means for different values of $k$.
2. For each value of $k$, calculate the silhouette score.
3. Plot the silhouette score as a function of $k$.
4. Look for the value of $k$ that maximizes the silhouette score.
```

<!--s-->

<div class="header-slide">

# Hierarchical Clustering

</div>

<!--s-->

## Hierarchical Clustering | Introduction

Hierarchical clustering builds a hierarchy of clusters. The hierarchy can be visualized as a dendrogram, which shows the relationships between clusters at different levels of granularity.

### Agglomerative Clustering
- Start with each data point as a separate cluster, and merge clusters iteratively. AKA "bottom-up" clustering.

### Divisive Clustering
- Start with all data points in a single cluster, and split clusters iteratively. AKA "top-down" clustering.

<!--s-->

## Hierarchical Clustering | Agglomerative Clustering

Agglomerative clustering is a bottom-up approach to clustering that starts with each data point as a separate cluster, and merges clusters iteratively based on a linkage criterion. The algorithm works as follows:

```text
1. Start with each data point as a separate cluster.
2. Compute the distance between all pairs of clusters.
3. Merge the two closest clusters.
4. Repeat steps 2-3 until the desired number of clusters is reached.
```
<img src = "https://www.knime.com/sites/default/files/public/6-what-is-clustering.gif" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Hayasaka, 2021</p>

<!--s-->

## Hierarchical Clustering | Agglomerative Clustering Linkage

The distance between clusters can be computed using different linkage methods, such as:

- **Single linkage** (minimum distance between points in different clusters)
- **Complete linkage** (maximum distance between points in different clusters)
- **Average linkage** (average distance between points in different clusters)

The choice of linkage method can have a significant impact on the clustering result.

<img src = "https://www.knime.com/sites/default/files/public/6-what-is-clustering.gif" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Hayasaka, 2021</p>

<!--s-->

## Hierarchical Clustering | Agglomerative Clustering

Agglomerative clustering is a flexible and intuitive algorithm, but it has some limitations:

- The algorithm is sensitive to the choice of distance metric and linkage method.
- The algorithm has a time complexity of $O(n^3)$, which can be slow for large datasets.

<!--s-->

<div class="header-slide">

# Density-Based Clustering

</div>

<!--s-->

## Density-Based Clustering | Introduction

Density-based clustering groups together points that are closely packed in the feature space. The idea is to identify regions of high density and separate them from regions of low density.

One popular density-based clustering algorithm is DBSCAN.

<img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*WunYbbKjzdXvw73a4Hd2ig.gif" height="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Seo, 2023</p>

<!--s-->

## Density-Based Clustering | DBSCAN

DBSCAN (Density-Based Spatial Clustering of Applications with Noise) is a density-based clustering algorithm that groups together points that are closely packed in the feature space. The algorithm works as follows:

```text
1. For a data point, determine the neighborhood of points within a specified radius $\epsilon$.
2. If the neighborhood contains at least $m$ points, mark the data point as a core point.
3. Expand the cluster by adding all reachable points to the cluster.
4. Repeat steps 1-3 until all points have been assigned to a cluster.
```

DBSCAN has two hyperparameters: $\epsilon$ (the radius of the neighborhood) and $m$ (the minimum number of points required to form a cluster). The algorithm is robust to noise and can identify clusters of arbitrary shape.

<!--s-->

## Density-Based Clustering | DBSCAN

DBSCAN is a powerful algorithm for clustering data, but it has some limitations:

- Sensitive to the choice of hyperparameters $\epsilon$ and $m$.
- May struggle with clusters of varying densities.


<!--s-->

## Summary

- Unsupervised machine learning is about finding patterns in $X$, without being given $y$.
- Clustering is a fundamental technique in unsupervised machine learning, with a wide range of applications.
    - **Partitional clustering** divides a dataset into $k$ clusters, with K-Means being a popular algorithm.
    - **Hierarchical clustering** builds a hierarchy of clusters, with agglomerative clustering being a common approach.
    - **Density-based clustering** groups together points that are closely packed in the feature space, with DBSCAN being a popular algorithm.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you working with **clustering** algorithms?

  Scan the QR code or go to [pollev.com/nucs](https://pollev.com/nucs)

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->