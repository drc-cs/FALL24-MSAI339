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
  ## L.07 | Supervised Machine Learning I
  ### KNN, Linear Regression, and Logistic Regression

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

- H.02 is due tonight @ 11:59 PM.
    - ~84% of you have already submitted! 🎉

- P.02 is due 10.31 or 11.05

<!--s-->


<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with the following methods:

  1. k-Nearest Neighbors (KNN)
  2. Linear Regression
  3. Logistic Regression with Gradient Descent

  Scan the QR code or go to [pollev.com/nucs](https://pollev.com/nucs)

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->

## L.07 | Supervised Machine Learning I

Today we will discuss several fundamental algorithms for **classification** and **regression**.

### Classification

Classification is a supervised learning task where the goal is to predict categorical class labels based on input features.

### Regression

Regression is a supervised learning task where the goal is to predict a continuous value based on the input features.

<!--s-->

## L.07 | Supervised Machine Learning I

Supervised machine learning tasks accept a feature matrix $X$ as input and predict a label $y$ as output. To achieve this aim, we will cover the following topics today:

- Splitting Data
  - Splitting data into training, validation, and testing sets.
- k-Nearest Neighbors (KNN)
  - Fundamental classification algorithm.
- Linear Regression
  - Fundamental regression algorithm.
- Logistic Regression
  - Extension of linear regression for classification.

<!--s-->

## Splitting Data into Training, Validation, and Testing Sets

Splitting data into training, validation, and testing sets is crucial for model evaluation and selection.

- **Training Set**: Used for fitting the model.
- **Validation Set**: Used for parameter tuning and model selection.
- **Test Set**: Used to evaluate the model performance.

A good, general rule of thumb is to split the data into 70% training, 15% validation, and 15% testing. In practice, k-fold cross-validation is often used to maximize the use of data. We will discuss this method in future lectures.

<img src="https://miro.medium.com/v2/resize:fit:1160/format:webp/1*OECM6SWmlhVzebmSuvMtBg.png" width="500" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Chavan, 2023</p>

<!--s-->

## Methods of Splitting

<div class = "col-wrapper">

<div class="c1" style = "width: 60%; font-size: 0.8em;">

### Random Split
Ensure you shuffle the data to avoid bias. Important when your dataset is ordered.

### Stratified Split
Used with imbalanced data to ensure each set reflects the overall distribution of the target variable. Important when your dataset has a class imbalance.

### Time-Based Split
Used for time series data to ensure the model is evaluated on future data. Important when your dataset is time-dependent.

### Group-Based Split
Used when data points are not independent, such as in medical studies. Important when your dataset has groups of related data points.

</div>

<div class="c2 col-centered" style = "width: 40%;">

```python
from sklearn.model_selection import train_test_split

X_train, X_temp, y_train, y_temp = train_test_split(X, y, test_size=0.3, random_state=42)
X_val, X_test, y_val, y_test = train_test_split(X_temp, y_temp, test_size=0.5, random_state=42)
```

<img src="https://miro.medium.com/v2/resize:fit:1160/format:webp/1*OECM6SWmlhVzebmSuvMtBg.png" width="500" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Chavan, 2023</p>

</div>

<!--s-->

<div class="header-slide">

# K-Nearest Neighbors (KNN)

</div>

<!--s-->

## k-Nearest Neighbors (KNN) | Concept

KNN is a non-parametric method used for classification (and regression!).

The principle behind nearest neighbor methods is to find a predefined number of samples closest in distance to the new point, and predict the label from these using majority vote.

<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/cs326-bucket/lecture_7/knn_setup.html"></iframe>

<!--s-->

## k-Nearest Neighbors (KNN) | What is it doing?

Given a new instance $ x' $, KNN classification computes the distance between $ x' $ and all other examples. The k closest points are selected and the predicted label is determined by majority vote.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; font-size: 0.7em;">

### Euclidean Distance

`$ d(x, x') =\sqrt{\sum_{i=1}^n (x_i - x'_i)^2} $`

### Manhattan Distance

`$ d(x, x') = \sum_{i=1}^n |x_i - x'_i| $`

### Cosine Distance 

`$ d(x, x') = 1 - \frac{x \cdot x'}{||x|| \cdot ||x'||} $`

### Jaccard Distance (useful for categorical data!)

`$ d(x, x') = 1 - \frac{|x \cap x'|}{|x \cup x'|} $`

### Hamming Distance (useful for strings!)

`$ d(x, x') = \frac{1}{n} \sum_{i=1}^n x_i \neq x'_i $`

</div>

<div class="c2 col-centered" style = "width: 60%;">

```python
from sklearn.neighbors import KNeighborsClassifier

# Default is Minkowski distance.
knn = KNeighborsClassifier(n_neighbors=3, metric='euclidean')
knn.fit(X_train, y_train)
```
</div>
</div>

<!--s-->

## k-Nearest Neighbors (KNN) | Example

Given the following data points (X) and their corresponding labels (y), what is the predicted label for the point (3.5, 3.5) using KNN with k=3?

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 50%">

<p style="margin-left: -10em;">A. 0<br><br>B. 1</p>

</div>
<div class="c2" style = "width: 50%">

<iframe width = "100%" height = "50%" src="https://storage.googleapis.com/cs326-bucket/lecture_7/knn_actual.html" padding=2em;></iframe>

```python
X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [0, 0, 1, 1]
knn = KNeighborsClassifier(n_neighbors=3)
knn.fit(X, y)
knn.predict([[3.5, 3.5]])
```

</div>
</div>

<!--s-->

## k-Nearest Neighbors (KNN) | Hyperparameters

### Number of Neighbors (k)

The number of neighbors to consider when making predictions.

### Distance Metric
The metric used to calculate the distance between points.

### Weights
Uniform weights give equal weight to all neighbors, while distance weights give more weight to closer neighbors.

### Algorithm
The algorithm used to compute the nearest neighbors. Some examples include Ball Tree, KD Tree, and Brute Force.

<!--s-->

## k-Nearest Neighbors (KNN) | Pros and Cons

### Pros

- Simple and easy to understand.
- No training phase.
- Can be used for both classification and regression.

### Cons

- Computationally expensive.
- Sensitive to the scale of the data.
- Requires a large amount of memory.

<!--s-->

## k-Nearest Neighbors (KNN) | Classification Model Evaluation

To evaluate a binary classification model like this, we can use metrics such as accuracy, precision, recall, F1 score, and ROC-AUC.


| Metric | Formula | Notes |
| --- | --- | --- | 
| Accuracy | $\frac{TP + TN}{TP + TN + FP + FN}$ | Easy to interpret but flawed.
| Precision | $\frac{TP}{TP + FP}$ | Useful when the cost of false positives is high. |
| Recall | $\frac{TP}{TP + FN}$ | Useful when the cost of false negatives is high. |
| F1 Score | $2 \times \frac{Precision \times Recall}{Precision + Recall}$ | Harmonic mean of precision and recall. | 
| ROC-AUC | Area under the ROC curve. | Useful for imbalanced datasets. |


<!--s-->

<div class="header-slide">

# Linear Regression

</div>

<!--s-->

## Linear Regression | Concept

Linear regression attempts to model the relationship between two or more variables by fitting a linear equation to observed data. The components to perform linear regression: 

$$ \widehat{y} = X\beta $$

Where $ \widehat{y} $ is the predicted value, $ X $ is the feature matrix, and $ \beta $ is the coefficient vector. The goal is to find the coefficients that minimize the error between the predicted value and the actual value.

<img src="http://www.stanford.edu/class/stats202/figs/Chapter3/3.1.png" width="600" style="margin: 0 auto; display: block;">

<!--s-->

## Linear Regression | Cost Function

The objective of linear regression is to minimize the cost function $ J(\beta) $:

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

$$ J(\beta) = \frac{1}{2m} \sum_{i=1}^m (\widehat{y}_i - y_i)^2 $$

Where $ \widehat{y} = X\beta $ is the prediction. This is most easily solved by finding the normal equation solution:

$$ \beta = (X^T X)^{-1} X^T y $$

The normal equation is derived by setting the gradient of $J(\beta) $ to zero. This is a closed-form solution that can be computed directly.

</div>
<div class="c2" style = "width: 50%">

```python
from sklearn.linear_model import LinearRegression

model = LinearRegression()
model.fit(X_train, y_train)
```

<img src="http://www.stanford.edu/class/stats202/figs/Chapter3/3.1.png" width="100%">

</div>
</div>

<!--s-->

## Linear Regression | Normal Equation Notes

### Adding a Bias Term

Practically, if we want to include a bias term in the model, we can add a column of ones to the feature matrix $ X $. Your H.03 will illustrate this concept.

$$ \widehat{y} = X\beta $$

### Gradient Descent

For large datasets, the normal equation can be computationally expensive. Instead, we can use gradient descent to minimize the cost function iteratively. We'll talk about gradient descent within the context of logistic regression later.

<!--s-->

## Linear Regression | Regression Model Evaluation

To evaluate a regression model, we can use metrics such as mean squared error (MSE), mean absolute error (MAE), mean absolute percentage error (MAPE), and R-squared.

<div style="font-size: 0.8em;">

**Mean Squared Error (MSE)**: The mean of the squared errors. Punishes large errors more than small errors.

$$ MSE = \frac{1}{m} \sum_{i=1}^m (\widehat{y}_i - y_i)^2 $$

**Mean Absolute Error (MAE)**: The mean of the absolute errors. Less sensitive to outliers than MSE.

$$ MAE = \frac{1}{m} \sum_{i=1}^m |\widehat{y}_i - y_i| $$

**Mean Absolute Percentage Error (MAPE)**: The mean of the absolute percentage errors. Useful for comparing models with different scales.

$$ MAPE = \frac{1}{m} \sum_{i=1}^m \left| \frac{\widehat{y}_i - y_i}{y_i} \right| \times 100 $$

**R-squared**: The proportion of the variance in the dependent variable that is predictable from the independent variables.

$$ R^2 = 1 - \frac{\sum(\widehat{y}_i - y_i)^2}{\sum(\bar{y} - y_i)^2} $$

</div>

<!--s-->

## Linear Regression | Pros and Cons

### Pros

- Simple and easy to understand.
- Fast to train.
- Provides a good, very interpretable baseline model.

### Cons

- Assumes a linear relationship between the features and the target variable.
- Sensitive to outliers.

<!--s-->

## Linear Regression | A Brief Note on Regularization

Regularization is a technique used to prevent overfitting by adding a penalty term to the cost function. The two most common types of regularization are L1 (Lasso) and L2 (Ridge) regularization.

**L1 Regularization**: Adds the absolute value of the coefficients to the cost function. This effectively performs feature selection by setting some coefficients to zero.

`$$ J(\beta) = \frac{1}{2m} \sum_{i=1}^m (\widehat{y}_i - y_i)^2 + \lambda \sum_{j=1}^n |\beta_j| $$`

**L2 Regularization**: Adds the square of the coefficients to the cost function. This shrinks the coefficients, but does not set them to zero. This is useful when all features are assumed to be relevant.

`$$ J(\beta) = \frac{1}{2m} \sum_{i=1}^m (\widehat{y}_i - y_i)^2 + \lambda \sum_{j=1}^n \beta_j^2 $$`

<!--s-->

<div class="header-slide">

# Logistic Regression

</div>


<!--s-->

## Logistic Regression | Concept

Logistic regression measures the relationship between the categorical dependent variable and one or more independent variables by estimating probabilities using a logistic function.

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*G3imr4PVeU1SPSsZLW9ghA.png" width="400" style="margin: 0 auto; display: block;">
<span style="font-size: 0.8em; text-align: center; display: block; color: grey;">Joshi, 2019</span>

<!--s-->

## Logistic Regression | Formula

This model is based on the sigmoid function $\sigma(z)$:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

Where 

$$ z = X\beta $$

Note that $\sigma(z)$ is the probability that the dependent variable is 1 given the input $X$. Consider the similar form of the linear regression model:

$$ \widehat{y} = X\beta $$

The key difference is that the output of logistic regression is passed through the sigmoid function to obtain the probability. This works because the sigmoid function maps any real number to the range [0, 1]. While linear regression predicts the value of the dependent variable, logistic regression predicts the probability that the dependent variable is 1.

<!--s-->

## Logistic Regression | No Closed-Form Solution

In linear regression, we can calculate the optimal coefficients $\beta$ directly. However, in logistic regression, we cannot do this due to the non-linearity (and thus, non-convexity) of the cost function.

Instead, we use gradient descent to minimize the cost function. Gradient descent is an optimization algorithm that iteratively updates the parameters to minimize the cost function, and forms the basis of many machine learning algorithms.

<img src="https://machinelearningspace.com/wp-content/uploads/2023/01/Gradient-Descent-Top2-1024x645.png" width="600" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Source: Medium</p>

<!--s-->

## Logistic Regression | Cost Function

The cost function used in logistic regression is the cross-entropy loss:

$$ J(\beta) = -\frac{1}{m} \sum_{i=1}^m [y_i \log(\widehat{y}_i) + (1 - y_i) \log(1 - \widehat{y}_i)] $$

$$ \widehat{y} = \sigma(X\beta) $$

Let's make sure we understand the intuition behind the cost function $J(\beta)$.

If the true label ($y$) is 1, we want the predicted probability ($\widehat{y}$) to be close to 1. If the true label ($y$) is 0, we want the predicted probability ($\widehat{y}$) to be close to 0. The cost goes up as the predicted probability diverges from the true label.

<!--s-->

## Logistic Regression | Gradient Descent

To minimize $ J(\beta) $, we update $ \beta $ iteratively using the gradient of $ J(\beta) $:

<div class = "col-wrapper">
<div class="c1" style = "width: 50%; font-size: 0.8em;">

$$ \beta := \beta - \alpha \frac{\partial J}{\partial \beta} $$

Where $ \alpha $ is the learning rate, and the gradient $ \frac{\partial J}{\partial \beta} $ is:

$$ \frac{\partial J}{\partial \beta} = \frac{1}{m} X^T (\sigma(X\beta) - y) $$

Where $ \sigma(X\beta) $ is the predicted probability, $ y $ is the true label, $ X $ is the feature matrix, $ m $ is the number of instances, $ \beta $ is the coefficient vector, and $ \alpha $ is the learning rate.

This is a simple concept that forms the basis of many gradient-based optimization algorithms, and is widely used in deep learning. 

Similar to linear regression -- if we want to include a bias term, we can add a column of ones to the feature matrix $ X $.

</div>
<div class="c2 col-centered" style = "width: 50%">

<img src="https://machinelearningspace.com/wp-content/uploads/2023/01/Gradient-Descent-Top2-1024x645.png" width="400" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Source: Medium</p>

</div>
</div>

<!--s-->

## Logistic Regression | Classifier

Once we have the optimal coefficients, we can use the logistic function to predict the probability that the dependent variable is 1. 

We can then use a threshold to classify the instance as 0 or 1 (usually 0.5). The following code snippet shows how to use the scikit-learn library to fit a logistic regression model and make predictions.


```python
from sklearn.linear_model import LogisticRegression

X = [[1, 2], [2, 3], [3, 4], [4, 5]]
y = [0, 0, 1, 1]
logistic_regression_model = LogisticRegression()
logistic_regression_model.fit(X_train, y_train)
logistic_regression_model.predict([[3.5, 3.5]])
```

```python
array([[0.3361201, 0.6638799]])
```

<!--s-->

## Question | Logistic Regression Output

Our logistic regression model was trained with $X = [[1, 2], [2, 3], [3, 4], [4, 5]]$ and $y = [0, 0, 1, 1]$. We then made a prediction for the point $[3.5, 3.5]$.

What does this output represent?

```python
array([[0.3361201, 0.6638799]])
```

<!--s-->

## Summary

- We discussed the importance of splitting data into training, validation, and test sets.
- We delved into k-Nearest Neighbors, Linear Regression, and Logistic Regression with Gradient Descent, exploring practical implementations and theoretical foundations.
- Understanding these foundational concepts is crucial for advanced machine learning and model fine-tuning! 

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Exit Poll
  ## On a scale of 1-5, how confident are you with the following methods:

  1. k-Nearest Neighbors (KNN)
  2. Linear Regression
  3. Logistic Regression with Gradient Descent

  Scan the QR code or go to [pollev.com/nucs](https://pollev.com/nucs)

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->