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
  ## L.08 | Supervised Machine Learning II

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

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  # Feedback
  ## What can we be doing to make this course better for **you**?
  Scan the QR code or go to [pollev.com/nucs](https://pollev.com/nucs)

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->

## Announcements

- H.03 will be released today.
  - Due on 10.24.2024. 

- P.02 is will take place on either 10.31 or 11.05.
  - Schedule posted at the end of the lecture.

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 60%; position: absolute;">

  # Intro Poll
  ## On a scale of 1-5, how confident are you with the following methods:

  1. ROC-AUC
  2. K-Fold Cross-Validation
  3. Multi-Class & Multi-Label Classification

  Scan the QR code or go to [pollev.com/nucs](https://pollev.com/nucs)

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->

## L.08 | Supervised Machine Learning II

Today we're going to discuss: 

1. ROC-AUC
2. K-Fold Cross-Validation
3. Multi-Class & Multi-Label Classification

<!--s-->

<div class="header-slide">

# ROC-AUC

</div>

<!--s-->

## ROC-AUC

- **Receiver Operating Characteristic (ROC) Curve**: A graphical representation of the performance of a binary classifier system as its discrimination threshold is varied.

- **Area Under the Curve (AUC)**: The area under the ROC curve, which quantifies the classifier’s ability to distinguish between classes.

<img src="https://assets-global.website-files.com/6266b596eef18c1931f938f9/64760779d5dc484958a3f917_classification_metrics_017-min.png" width="400" style="margin: 0 auto; display: block;border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Source: Evidently AI</p>

<!--s-->

## ROC-AUC | Key Concepts

<div style="font-size: 0.7em;">

**True Positive Rate (TPR)**: The proportion of actual positive cases that are correctly identified by the classifier.<br>
**False Positive Rate (FPR)**: The proportion of actual negative cases that are incorrectly identified as positive by the classifier.<br>
**ROC**: When the FPR is plotted against the TPR for each binary classification threshold, we obtain the ROC curve.
</div>
<img src="https://miro.medium.com/v2/resize:fit:4800/format:webp/1*CQ-1ceyX80EE0a_s3SwvgQ.png" width="100%" style="margin: 0 auto; display: block;border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Cortex, 2020</p>

<!--s-->

## ROC-AUC | Key Concepts

<img src="https://miro.medium.com/v2/resize:fit:4512/format:webp/1*zNtuQziwUKkGUxhG0Go5kA.png" width="100%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Cortex, 2020</p>

<!--s-->

## Question | ROC-AUC

I have a binary classifier that predicts whether a patient has a disease. The ROC-AUC of the classifier is 0.4. What does this mean?

<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
&emsp;A. The classifier is worse than random. <br>
&emsp;B. The classifier is random. <br>
&emsp;C. The classifier is better than random. <br>
</div>
</div>
<div class='c2' style = 'width: 40%; display: flex; align-items: center; flex-direction: column;'>
<img src='https://storage.googleapis.com/slide_assets/PollEverywhere.png' width='100%'>
<a>poll.ev.com/nucs</a>
</div>
</div>

<!--s-->

<div class="header-slide">

# K-Fold Cross-Validation

</div>

<!--s-->

## K-Fold Cross-Validation

K-Fold Cross-Validation is a technique used to evaluate the performance of a machine learning model. It involves splitting the data into K equal-sized folds, training the model on K-1 folds, and evaluating the model on the remaining fold. This process is repeated K times, with each fold serving as the validation set once.

<img src="https://miro.medium.com/v2/resize:fit:1400/format:webp/1*AAwIlHM8TpAVe4l2FihNUQ.png" width="100%" style="margin: 0 auto; display: block;border-radius: 10px;">
<span style="font-size: 0.8em; text-align: center; display: block; color: grey;">Patro 2021</span>

<!--s-->

## K-Fold Cross-Validation | Advantages

When implemented correctly, K-Fold Cross-Validation has several advantages:

- **Better Use of Data**: K-Fold Cross-Validation uses all the data for training and validation, which can lead to more accurate estimates of model performance.

- **Reduced Variance**: By averaging the results of K different validation sets, K-Fold Cross-Validation can reduce the variance of the model evaluation.

- **Model Selection**: K-Fold Cross-Validation can be used to select the best model hyperparameters.

<!--s-->

<div class="header-slide">

# Multi-Class & Multi-Label Classification

</div>

<!--s-->

## Multi-Class Classification vs Multi-Label Classification

- **Multi-Class Classification**: A classification task with more than two classes, where each sample is assigned to one class.

- **Multi-Label Classification**: A classification task where each sample can be assigned to multiple classes.

<img src ="https://aman.ai/primers/ai/assets/multiclass-vs-multilabel-classification/mc-vs-ml.jpeg" width="80%" style="margin: 0 auto; display: block; border-radius: 10px;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Aman Chadha, 2020</p>

<!--s-->

## Multi-Class Classification | Logistic Regression

In multi-class classification, each sample is assigned to one class. The goal is to predict the class label of a given sample. We've already seen at least one model that can be used for multi-class classification: Logistic Regression!

Logistic regression can be extended to multi-class classification using the softmax function.

<img src="https://www.researchgate.net/publication/342987800/figure/fig1/AS:913942624870401@1594912310090/Binary-vs-Multiclass-classification.jpg" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Emerson</p>

<!--s-->

## Multi-Class Classification | Updating Logistic Regression

To use Logistic Regression for multi-class classification, we need to update the model. Recall that the logistic function is defined as:

$$ \sigma(z) = \frac{1}{1 + e^{-z}} $$

For multi-class classification, we use the softmax function instead:

$$ \sigma(z_j) = \frac{e^{z_j}}{\sum_{k=1}^{K} e^{z_k}} $$

Where:

- $z_j$ is the input to the $j$-th output unit.
- $K$ is the number of classes.

Intuitively, the softmax function converts the output of the model into probabilities for each class, with a sum of 1. An output may look like: 

<span class="code-span">$$[0.1, 0.6, 0.3]$$</span>

<!--s-->

## Multi-Class Classification | Updating Logistic Regression

Recall binary cross-entropy:

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} y^{(i)} \log(\widehat{y}^{(i)}) + (1 - y^{(i)}) \log(1 - \widehat{y}^{(i)}) $$

Alternatively, the cost function for multi-class classification is the cross-entropy loss:

$$ J(\theta) = -\frac{1}{m} \sum_{i=1}^{m} \sum_{k=1}^{K} y_k^{(i)} \log(\widehat{y}_k^{(i)}) $$


Where:

- $m$ is the number of samples.
- $K$ is the number of classes.
- $y_k^{(i)}$ is the true label of the $i$-th sample for class $k$.
- $\widehat{y}_k^{(i)}$ is the predicted probability of the $i$-th sample for class $k$.

<!--s-->

## Multi-Class Classification | Updating Logistic Regression

| Binary Classification | Multi-Class Classification |
|-----------------------|-----------------------------|
| Sigmoid Function      | Softmax Function            |
| Binary Cross-Entropy  | Cross-Entropy Loss          |
| One output unit, one class. The negative class is just 1 - positive class. | $K$ output units, $K$ classes  with probabilities that sum to 1. |
| One weight vector | $K$ weight vectors |

You don't need to implement the softmax function or cross-entropy loss from scratch for this course, but it's important to understand the intuition behind them!

<!--s-->

## Multi-Class Classification | Other Approaches

There are other approaches to multi-class classification, such as:

- **One-vs-All (OvA)**: Train $K$ binary classifiers, one for each class. During prediction, choose the class with the highest probability.

- **One-vs-One (OvO)**: Train $K(K-1)/2$ binary classifiers, one for each pair of classes. During prediction, choose the class with the most votes.

<img src="https://dezyre.gumlet.io/images/blog/multi-class-classification-python-example/image_504965436171642418833831.png?w=1100&dpr=2.0" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">ProjectPro</p>

<!--s-->

## Multi-Class Classification | One-vs-All

In One-vs-All (OvA), we train $K$ binary classifiers, one for each class. During prediction, we choose the class with the highest probability.

For example, if we have three classes (A, B, C), we would train three binary classifiers:

- Classifier 1: A vs. (B, C)
- Classifier 2: B vs. (A, C)
- Classifier 3: C vs. (A, B)

<img src="https://dezyre.gumlet.io/images/blog/multi-class-classification-python-example/image_504965436171642418833831.png?w=1100&dpr=2.0" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">ProjectPro</p>

<!--s-->

## Multi-Class Classification | One-vs-One

In One-vs-One (OvO), we train $K(K-1)/2$ binary classifiers, one for each pair of classes. During prediction, we choose the class with the most votes.

For example, if we have three classes (A, B, C), we would train three binary classifiers:

- Classifier 1: A vs. B
- Classifier 2: A vs. C
- Classifier 3: B vs. C

<img src="https://dezyre.gumlet.io/images/blog/multi-class-classification-python-example/image_504965436171642418833831.png?w=1100&dpr=2.0" width="50%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">ProjectPro</p>

<!--s-->

## Multi-Label Classification

In multi-label classification, each sample can be assigned to multiple classes. Some examples include:

- Tagging images with multiple labels.
- Predicting the genre of a movie.
- Classifying text into multiple categories.

<img src ="https://media.licdn.com/dms/image/v2/D4D12AQGGdfJ43myRIw/article-cover_image-shrink_720_1280/article-cover_image-shrink_720_1280/0/1662243329674?e=1733961600&v=beta&t=MtXglqFR_iBh5NfnTswJtYwD-_wprTxgu0K2aHuYlh4" width="70%" style="margin: 0 auto; display: block;">
<p style="text-align: center; font-size: 0.6em; color: grey;">Hesham, 2022</p>

<!--s-->

## Multi-Label Classification | Approaches

There are several approaches to multi-label classification:

**Binary Relevance**: Train a separate binary classifier for each label. 
- e.g. one classifier for each genre of a movie.

**Classifier Chains**: Train a chain of binary classifiers, where each classifier predicts the next label.
- e.g. predict the next genre of a movie based on the previous genre.

**Label Powerset**: Treat each unique label combination as a single class. 
- e.g. treat the combination of genres as a single class.

<!--s-->

## Multi-Label Classification | Comparisons

| Approach | Pros | Cons |
|----------|------|------|
| Binary Relevance | Simple, easy to implement | Ignores label dependencies |
| Classifier Chains | Considers label dependencies | Sensitive to label order |
| Label Powerset | Considers label dependencies | Exponential increase in classes |

<!--s-->

## Question | Multi-Class vs Multi-Label

Let's say we have a dataset of movie genres. We want to predict the genre(s) of a movie. Should we use multi-class or multi-label classification?

<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
&emsp;A. Multi-Class <br>
&emsp;B. Multi-Label <br>
</div>
</div>
<div class='c2' style = 'width: 40%; display: flex; align-items: center; flex-direction: column;'>
<img src='https://storage.googleapis.com/slide_assets/PollEverywhere.png' width='100%'>
<a>poll.ev.com/nucs</a>
</div>
</div>

<!--s-->

## Question | Multi-Class vs Multi-Label

Let's say we have a dataset of movie genres. We know that a movie can only one genre, but there are many to choose from. Should we use multi-class or multi-label classification?

<div class='col-wrapper' style = 'display: flex; align-items: top; margin-top: 2em; margin-left: -1em;'>
<div class='c1' style = 'width: 60%; display: flex; align-items: center; flex-direction: column; margin-top: 2em'>
<div style = 'line-height: 2em;'>
&emsp;A. Multi-Class <br>
&emsp;B. Multi-Label <br>
</div>
</div>
<div class='c2' style = 'width: 40%; display: flex; align-items: center; flex-direction: column;'>
<img src='https://storage.googleapis.com/slide_assets/PollEverywhere.png' width='100%'>
<a>poll.ev.com/nucs</a>
</div>
</div>

<!--s-->

## Key takeaways:

### ROC-AUC
A key metric for evaluating binary classifiers.
### K-Fold Cross-Validation
A technique for evaluating machine learning models, and makes better use of the data.
### Multi-Class Classification
Involves predicting the class label of a given sample when there are more than two classes.
### Multi-Label Classification
Involves predicting multiple labels for a given sample.

<!--s-->

## P.02 (Elevator Pitch)

<div class = "col-wrapper">
<div class="c1" style = "width: 50%;">

### 10.31.2024

- Crispy
- DataCats
- Gryffindor
- HPS

</div>

<div class="c2" style = "width: 50%;">

### 11.05.2024

- JDNH
- One two three
- Outliers
- Powerpuff Girls

</div>
</div>

<!--s-->

<div class="header-slide">

# H.03 | machine_learning.py

<iframe src="https://lottie.host/embed/6c677caa-d54a-411c-b0c0-6f186378d571/UKVVhf0EJN.json" height = 200></iframe>

</div>

<!--s-->