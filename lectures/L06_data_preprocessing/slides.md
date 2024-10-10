---
title: MSAI 339
separator: <!--s-->
verticalSeparator: <!--s-->
theme: serif
revealOptions:
  transition: 'none'
---

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 70%; position: absolute;">

  # Data Science
  ## L.06 | Data Preprocessing

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 80%; padding-top: 30%">
  
  <iframe src="https://lottie.host/embed/bd6c5b65-d724-4f97-882c-40f58367ea38/BIKhZdSeqW.json" height="100%" width = "100%"></iframe>
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

- H.02 is due 10.15

- P.02 is due 10.31 or 11.05

<!--s-->

<div class="header-slide">

# Data Preprocessing

</div>

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you in taking a raw dataset and transforming it into a machine learning-ready dataset?

  Scan the QR code or go to [pollev.com/nucs](https://pollev.com/nucs)

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->

## Common Data Format

Often, your goal is to build a complete, numerical matrix with rows (instances) and columns (features).

- **Ideal**: Raw Dataset → Model → Task
- **Reality**: Raw Dataset → ML-Ready Dataset → Features (?) → Model → Task

Once you have a numerical matrix, you can apply modeling methods to it.

| Student ID | Homework | Exam 1 | Exam 2 | Final |
|------------|----------|--------|--------|-------|
| [1,0,...] | 0.90  | 0.89 | 0.70 | 0.93 |
| [0,1,...] | 0.79 | 0.75| 0.70 | 0.65 |
| ... | ... | ... | ... | ... |

<!--s-->

## Overview | From Raw Dataset to ML-Ready Dataset

The raw dataset is transformed into a machine learning-ready dataset by converting the data into a format that can be used by machine learning models.

<div style = "font-size: 0.8em;">

### Raw Dataset:

| Student ID | Homework | Exam 1 | Exam 2 | Final |
|------------|----------|--------|--------|-------|
| "2549@nw.edu" | 90%| 89% | None | 93% |
| "7856@nw.edu" | 79%| 75% | 70% | 65% |
| ... | ... | ... | ... | ... |

### Processed Dataset:

| Student ID | Homework | Exam 1 | Exam 2 | Final |
|------------|----------|--------|--------|-------|
| [1,0,...] | 0.90  | 0.89 | 0.70 | 0.93 |
| [0,1,...] | 0.79 | 0.75| 0.70 | 0.65 |
| ... | ... | ... | ... | ... |

### ML-Ready Dataset ($X$ & $y$):

<div class = "col-centered">
[<br>
&emsp;[1, 0, ..., 0.90, 0.89, 0.70, 0.93],<br>
&emsp;[0, 1, ..., 0.79, 0.75, 0.70, 0.65],<br>
]
</div>

<!--s-->

## Overview | Adding Feature Engineering

Feature engineering is the process of using domain knowledge to select and transform raw data into features that increase the predictive power of the learning models.

You can often consider feature engineering as a way to add new columns to your dataset that may help the model better understand the data (i.e. spoon-feeding a model with your domain knowledge). <br><br><br>

<div class = "col-wrapper">
<div class="c1" style = "width: 50%;  font-size: 0.8em;">

### Processed:

| Student ID | Homework | Exam 1 | Exam 2 | Final |
|------------|----------|--------|--------|-------|
| [1,0,...] | 0.90  | 0.89 | 0.70 | 0.79 |
| [0,1,...] | 0.79 | 0.75| 0.70 | 0.73 |
| ... | ... | ... | ... | ... |


</div>
<div class="c2" style = "width: 50%;  font-size: 0.8em;">

### Processed + Engineered Feature:

| Student ID | Homework | Exam 1 | Exam 2 | Average Exam Score | Final |
|------------|----------|--------|--------|---------|-------|
| [1,0,...] | 0.90  | 0.89 | 0.70 | 0.79 | 0.93 |
| [0,1,...] | 0.79 | 0.75| 0.70 | 0.74 | 0.65 |
| ... | ... | ... | ... | ... | ... |

</div>
</div>

<!--s-->

## Overview | Interaction Terms with PolynomialFeatures

PolynomialFeatures generates a new feature matrix consisting of all polynomial combinations of the features with degree less than or equal to the specified degree. For example, if an input sample is two dimensional and of the form 

$$[a, b]$$

the degree-2 polynomial features are 

$$ [1, a, b, a^2, ab, b^2] $$

this is a very quick way to add interaction terms to your dataset.

```python
import numpy as np
from sklearn.preprocessing import PolynomialFeatures
X = np.arange(6).reshape(3, 2)
poly = PolynomialFeatures(degree=2, interaction_only=False,  include_bias=True)
poly.fit_transform(X)
```

```
# Original
[[0 1]
 [2 3]
 [4 5]]
```

```
# Transformed
array([[ 1.,  0.,  1.,  0.,  0.,  1.],
       [ 1.,  2.,  3.,  4.,  6.,  9.],
       [ 1.,  4.,  5., 16., 20., 25.]])
```

<!--s-->

## Data Transformations

Today we will cover a number of transformations and feature engineering methods that are common in the data preprocessing stage.

<div style="font-size: 0.75em">

| Type | Transformation | Description |
|-----------|----------------|-------------|
| Numerical | Binarization | Convert numeric to binary. |
| Numerical | Binning | Group numeric values. |
| Numerical | Log Transformation | Manage data scale disparities. |
| Numerical | Scaling | Standardize or scale features. |
| Categorical | One-Hot Encoding | Convert categories to binary columns. |
| Categorical | Feature Hashing | Compress categories into hash vectors. |
| Temporal | Temporal Features | Convert to bins, manage time zones. |
| Missing | Imputation | Fill missing values. [L.02](https://drc-cs.github.io/cs326/lectures/L02_data_sources/#/25).|
| High-Dimensional | Feature Selection | Choose relevant features. |
| High-Dimensional | Feature Sampling | Reduce feature space via feature sampling. |
| High-Dimensional | Random Projection | Reduce feature space via random projection. |
| High-Dimensional | Principal Component Analysis (PCA) | Reduce dimensions while preserving data variability. |
</div>

<p style="text-align: left; font-size: 0.8em;">* Text data will be covered during the NLP lectures</p>

<!--s-->

## Numerical Data | Binarization

Convert numerical values to binary values via a threshold. Values above the threshold are set to 1, below threshold are set to 0.

```python
from sklearn.preprocessing import Binarizer

transformer = Binarizer(threshold=3).fit(data)
transformer.transform(data)
```

<img src="https://storage.googleapis.com/cs326-bucket/lecture_6/binarized.png" width="800" style="display: block; margin: 0 auto; border-radius: 10px;">

<!--s-->

## Numerical Data | Binning

Group numerical values into bins.

- **Uniform**: Equal width bins. Use this when the data is uniformly distributed.
- **Quantile**: Equal frequency bins. Use this when the data is not uniformly distributed, or when you want to be robust to outliers.

```python
from sklearn.preprocessing import KBinsDiscretizer

# uniform binning
transformer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='uniform').fit(data)
transformer.transform(data)

# quantile binning
transformer = KBinsDiscretizer(n_bins=5, encode='ordinal', strategy='quantile').fit(data)
transformer.transform(data)
```

<!--s-->

## Numerical Data | Binning

<img src="https://storage.googleapis.com/cs326-bucket/lecture_6/binning.png">

<!--s-->

## Numerical Data | Log Transformation

Logarithmic transformation of numerical values. This is useful for data with long-tailed distributions, or when the scale of the data varies significantly.


```python
import numpy as np
transformed = np.log(data)
```

<br><br>

<img src="https://www.researchgate.net/profile/Matthieu-Komorowski-2/publication/308007227/figure/fig5/AS:405478883512321@1473685107077/55-Example-of-the-effect-of-a-log-transformation-on-the-distribution-of-the-dataset.png" width="500" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Komorowski, 2016</p>


<!--s-->

## Numerical Data | Scaling

Standardize or scale numerical features.

- **MinMax**: Squeeze values into [0, 1].

$$ x_{\text{scaled}} = \frac{x - \text{min}(x)}{\text{max}(x) - \text{min}(x)} $$

- **Standard**: Standardize features to have zero mean and unit variance:

$$ x_{\text{scaled}} = \frac{x - \mu}{\sigma} $$

```python
from sklearn.preprocessing import MinMaxScaler, StandardScaler

# min-max scaling
scaler = MinMaxScaler().fit(data)
scaler.transform(data)

# standard scaling
scaler = StandardScaler().fit(data)
scaler.transform(data)
```

<!--s-->

## Numerical Data | Scaling

| Method | Pros | Cons | When to Use | 
|--------|------|------| ------------|
| MinMax | Preserves relationships. Bounded between 0 and 1. | Sensitive to outliers | When the data is uniformly distributed. |
| Standard | More robust to outliers. | May not preserve relationships. Not bounded. | When the data is normally distributed. |

**A special note on training / testing sets** -- always fit the scaler on the training set and transform both the training and testing sets using those parameters. This ensures that the testing set is not used to influence the training set.

<!--s-->

## Numerical Data | Why Scale Data?

### Convergence
Some models (like SVMs and neural networks) converge faster when the data is scaled. Scaling the data can help the model find the optimal solution more quickly.

### Performance
Some models (like KNN) are sensitive to the scale of the data. Scaling the data can improve the performance of these models. Consider the KNN model -- if the data is not scaled, the model will give more weight to features with larger scales.

### Regularization
Regularization methods (like L1 and L2 regularization) penalize large coefficients. If the features are on different scales, the regularization term may penalize some features more than others.

<!--s-->

## Categorical Data | One-Hot Encoding

Convert categorical features to binary columns. This will create a binary column for each unique value in the data. For example, the color feature will be transformed into three columns: red, green, and blue.

```python
from sklearn.preprocessing import OneHotEncoder

data = [["red"], ["green"], ["blue"]]
encoder = OneHotEncoder().fit(data)
encoder.transform(data)
```


```
array([[0., 0., 1.],
       [0., 1., 0.],
       [1., 0., 0.]])
```

<!--s-->

## Categorical Data | Feature Hashing

Compress categorical features into hash vectors -- in other words, this method reduces the dimensionality of the feature space. Hashing is using a function that maps categorical values to fixed-length vectors.

Compared to one-hot encoding, feature hashing is more memory-efficient and can handle high-dimensional data. It is typically used when some categories are unknown ahead of time, or when the number of categories is very large.

However, it can lead to collisions, where different categories are mapped to the same column. In this case below, we are hashing the color feature into a 2-dimensional vector. 


```python
from sklearn.feature_extraction import FeatureHasher

data = [{"color": "red"}, {"color": "green"}, {"color": "blue"}]
hasher = FeatureHasher(n_features=2, input_type='dict').fit(data)
hasher.transform(data)
```

```
array([[-1,  0],
       [ 0,  1],
       [ 1,  0]])
```

<!--s-->

## Question | One-Hot Encoding vs. Feature Hashing

You are working with a dataset that contains a feature with 100 unique categories. You are unsure if all categories are present in the training set, and you want to reduce the dimensionality of the feature space. Which method would you use?

<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
&emsp;A. One-Hot Encoding <br>
&emsp;B. Feature Hashing <br>
</div>
</div>
<div class='c2' style = 'width: 40%; display: flex; align-items: center; flex-direction: column;'>
<img src='https://storage.googleapis.com/slide_assets/PollEverywhere.png' width='100%'>
<a>poll.ev.com/nucs</a>
</div>
</div>

<!--s-->

## Temporal Data | Temporal Granularity

Converting temporal features to bins based on the required granularity for your model helps manage your time series data. For example, you can convert a date feature to year / month / day, depending on the required resolution of your model.

We'll cover more of time series data handling in our time series analysis lecture -- but pandas has some great tools for this!

```python
import pandas as pd

data = pd.DataFrame({"date": ["2021-01-01", "2021-02-01", "2021-03-01"]})
data["date"] = pd.to_datetime(data["date"])
data["month"] = data["date"].dt.month
data["day"] = data["date"].dt.day

```

date|month|day
---|---|---
2021-01-01|1|1
2021-02-01|2|1
2021-03-01|3|1

<!--s-->

## Missing Data | Imputation

We covered imputation in more detail in [L.02](https://drc-cs.github.io/cs326/lectures/L02_data_sources/#/25).

<div style = "font-size: 0.85em;">


| Method | Description | When to Use |
| --- | --- | --- |
| Forward / backward fill | Fill missing value using the last / next valid value. | Time Series |
| Imputation by interpolation | Use interpolation to estimate missing values. | Time Series |
| Mean value imputation | Fill missing value with mean from column. | Random missing values |
| Conditional mean imputation | Estimate mean from other variables in the dataset. | Random missing values |
| Random imputation | Sample random values from a column. | Random missing values | 
| KNN imputation | Use K-nearest neighbors to fill missing values. | Random missing values |
| Multiple Imputation | Uses many regression models and other variables to fill missing values. | Random missing values |

</div>

<!--s-->

<div class="header-slide">

# Dimensionality Reduction

</div>

<!--s-->

## High-Dimensional Data | Why Reduce Dimensions?

- **Curse of Dimensionality**: As the number of features increases, the amount of data required to cover the feature space grows exponentially.

- **Overfitting**: High-dimensional data is more likely to overfit the model, leading to poor generalization.

- **Computational Complexity**: High-dimensional data requires more computational resources to process.

- **Interpretability**: High-dimensional data can be difficult to interpret and visualize.

<!--s-->

## High-Dimensional Data | The Curse of Dimensionality

**tldr;** As the number of features increases, the amount of data required to cover the feature space grows exponentially. This can lead to overfitting and poor generalization.

**Intuition**: Consider a 2D space with a unit square. If we divide the square into 10 equal parts along each axis, we get 100 smaller squares. If we divide it into 100 equal parts along each axis, we get 10,000 smaller squares. The number of smaller squares grows exponentially with the number of divisions. Without exponentially growing data points for these smaller squares, a model needs to make more and more inferences about the data.

**Takeaway**: With regards to machine learning, this means that as the number of features increases, the amount of data required to cover the feature space grows exponentially. Given that we need more data to cover the feature space effectively, and we rarely do, this can lead to overfitting and poor generalization.

<img src="https://www.researchgate.net/profile/Mahua-Bhattacharya/publication/264823819/figure/fig1/AS:651533758787592@1532349161929/The-curse-of-dimensionality-a-11-objects-in-one-unit-bin-b-6-objects-in-one-unit-bin.png" width="500" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Rajput, 2012</p>

<!--s-->

## High-Dimensional Data | Feature Selection

One method to reduce dimensionality is to choose relevant features based on their importance for classification. The example below uses the chi-squared test to select the 20 best features. 

This works by selecting the features that are least likely to be independent of the class label (i.e. the features that are most likely to be relevant for classification).


```python
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectKBest, chi2

X, y = load_digits(return_X_y=True)
X_new = SelectKBest(chi2, k=20).fit_transform(X, y)
```

<!--s-->

## High-Dimensional Data | Feature Sampling

Another way is to reduce the feature space using modeling methods. The example below uses a random forest classifier to select the most important features. 

This works because the random forest model is selecting the features that are most likely to be important for classification. We'll cover random forests in more detail in a future lecture.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<img src="https://miro.medium.com/v2/resize:fit:1010/1*R3oJiyaQwyLUyLZL-scDpw.png" width="300" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Deniz Gunay, 2023</p>

</div>
<div class="c2" style = "width: 70%">

```python
from sklearn.datasets import load_digits
from sklearn.feature_selection import SelectFromModel
from sklearn.ensemble import RandomForestClassifier

X, y = load_digits(return_X_y=True)
X_new = SelectFromModel(RandomForestClassifier()).fit_transform(X, y)
```

</div>
</div>


<!--s-->

## High-Dimensional Data | Random Projection

Gaussian random projection is a type of random projection that uses a Gaussian distribution to generate the random matrix. It takes the original data and projects it into a lower-dimensional space.

Johnson-Lindenstrauss Lemma states that a random projection can preserve the pairwise distances between the data points with high probability (up to a constant factor).

<img src = "https://www.researchgate.net/publication/362583995/figure/fig3/AS:11431281085524461@1663781154442/Dimension-Reduction-using-Random-Projection-37.jpg" width="500" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Ratra, 2022</p>

```python
from sklearn.random_projection import GaussianRandomProjection

transformer = GaussianRandomProjection(n_components=2).fit(data)
transformer.transform(data)
```

<!--s-->

## High-Dimensional Data | Principal Component Analysis (PCA)

PCA reduces dimensions while preserving data variability. PCA works by finding the principal components of the data, which are the directions in which the data varies the most. It then projects the data onto these principal components, reducing the dimensionality of the data while preserving as much of the variability as possible.

<img src = "https://numxl.com/wp-content/uploads/principal-component-analysis-pca-featured.png" width="500" style="display: block; margin: 0 auto;">
<p style="text-align: center; font-size: 0.6em; color: grey;">NumXL</p>

```python
from sklearn.decomposition import PCA

pca = PCA(n_components=2)
pca.fit(data)
pca.transform(data)
```

<!--s-->

## Summary

- 🔥 A common mistake is standardizing the entire dataset before splitting it into training and testing sets. Always fit the scaler on the training set and transform both the training and testing sets using those parameters.

- There are a number of ways that you can transform your data, regardless of type, to make it more suitable for machine learning models. Having more transformations in your toolbox can help you better understand your data and build better models.

- There is no one-size-fits-all solution to data preprocessing. The methods you choose will depend on the data you have and the model you are building.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you in taking a raw dataset and transforming it into a machine learning-ready dataset?

  Scan the QR code or go to [pollev.com/nucs](https://pollev.com/nucs)

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->

<div class="header-slide">

# P.02 | Elevator Pitch

</div>

<!--s-->

## P.02 | Elevator Pitch

An elevator pitch is a brief, persuasive speech that you can use to spark interest in what your project is about. You can also use it to create interest in a project, idea, or product – or in yourself. A good elevator pitch should last no longer than a short elevator ride of 20 to 30 seconds, hence the name.

For P.02, you are expected to do an elevator pitch style presentation. This means you should be able to explain your project in a concise and engaging manner. We'll relax the contraints a bit, but you should aim for a total 5 minute presentation that at minimum covers your plans for the following aspects of your project:

1. Data Source
2. Data Exploration
3. Data Preprocessing
4. Data Modeling
5. Data Interpretation

You should have **one slide per topic**. Everyone needs to speak. You will each be graded on your ability to communicate your domain of the project effectively and concisely. Your team will have this presentation on 10.31 or 11.05.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 1em; left: 0; width: 50%; position: absolute;">

  # H.02 | basic_statistics.py
  ## Code & Free Response
  ### Due: 10.15.2024 @ 11:59 PM

```bash
cd <class directory>
git pull
```

</div>
</div>
<div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 0%">
<iframe src="https://lottie.host/embed/6c677caa-d54a-411c-b0c0-6f186378d571/UKVVhf0EJN.json" height = "100%" width = "100%"></iframe>

</div>
</div>

