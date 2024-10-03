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
  ## L.04 | Correlation & Association

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

- **P.01** is due 10.08.2024 at 11:59 PM.

- **GA Office Hours**:
    - Wednesdays from 1-2PM in Annenberg G28.

- **Office Hours**:
    - Send me an email -- I'm on campus Tuesdays and Thursdays. 

<!--s-->

<div class = "header-slide">

# Correlation

</div>

<!--s-->

## Relationships in Data

Our goal is often to understand the relationships between different variables in our data.

- **Correlation**: Quantitatively measures the degree to which two variables move in relation to each other.

- **Association**: Describes the relationship between two variables in terms of co-occurrence.

<!--s-->

## Correlation

Correlation is a fundamental concept in data science. It is used to measure the strength and direction of a relationship between two variables.

<div style="text-align: center; height: 60%">
  <img src="https://www.scribbr.com/wp-content/uploads/2022/07/Perfect-positive-correlation-Perfect-negative-correlation.webp">
</div>
<!--s-->

## Correlation | Quantitative Measurement via Covariance

**Covariance** is a measure of how much two random variables vary together, which is a measure of their **correlation**.

The covariance between two variables \(X\) and \(Y\) can be defined as:

<div class="col-centered">
$ \text{cov}(X, Y) = \frac{\sum_{i=1}^n (x_i - \bar{x})(y_i - \bar{y})}{n} $
</div>

And it's really just a generalization of variance (one variable):

<div class="col-centered">

$ \sigma^2 = \frac{\sum_{i=1}^n (x_i - \bar{x})^2}{n} $

</div>

<!--s-->

## Correlation | Interpreting Covariance

When the covariance is positive, it means that the two variables are moving in the same direction. When the covariance is negative, it means that the two variables are moving in opposite directions.

**But** size of the covariance is not standardized, so it is difficult to interpret the strength of the relationship. Consider the following example:

**Case 1:**
- **Study Hours (X):** <span class="code-span">[5, 10, 15, 20, 25]</span>
- **Test Scores (Y):** <span class="code-span">[50, 60, 70, 80, 90]</span>

**Case 2:**
- **Study Hours (X):** <span class="code-span">[5, 10, 15, 20, 25]</span>
- **Test Scores (Y):** <span class="code-span">[500, 600, 700, 800, 900]</span>

Covariance will be different in these cases, but the relationship is the same!

<!--s-->

## Correlation | Pearson Correlation Coefficient

Pearson correlation coefficient, denoted by \(r\), is a measure of the linear correlation between two variables. It ranges from -1 to 1, and so it is a **standardized** measure of the strength of the relationship.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

$r = \frac{\sum_i (x_i - \bar{x})(y_i - \bar{y})}{\sqrt{\sum_i (x_i - \bar{x})^2}\sqrt{\sum_i (y_i - \bar{y})^2}} $

<span class="code-span">r = 1</span>: Perfect positive linear relationship <br>
<span class="code-span">r = -1</span>: Perfect negative linear relationship <br>
<span class="code-span">r = 0</span>: No linear relationship



</div>
<div class="c2" style = "width: 50%">

<img src="https://www.scribbr.com/wp-content/uploads/2022/07/Perfect-positive-correlation-Perfect-negative-correlation.webp">

</div>
</div>

<!--s-->

## Correlation | Pearson Correlation Coefficient

Pearson's correlation coefficient is a great method to measure the strength of a linear relationship between two variables. However, it has some limitations:

- Sensitive to outliers.
- It only measures linear relationships.

If your relationship is not linear, or you have big outliers, you may want to consider another correlation method (e.g., Spearman's rank correlation coefficient).

<!--s-->

## Correlation | Spearman Rank Correlation Coefficient

Spearman Rank Correlation Coefficient counts the number of disordered pairs, not how well the data fits a line. Thus, it is better for non-linear relationships. You can use the formula below only if all n ranks are distinct integers.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<div>
$ r_s = 1 - \frac{6 \sum_i d_i^2}{n^3 - n} $
</div>
<div>
$ d_i = \text{rank}(x_i) - \text{rank}(y_i) $
</div>


<span class="code-span">r_s = 1</span>: Perfect positive relationship <br>
<span class="code-span">r_s = -1</span>: Perfect negative relationship <br>
<span class="code-span">r_s = 0</span>: No relationship

</div>
<div class="c2" style = "width: 50%">

<img src="https://www.scribbr.com/wp-content/uploads/2021/08/monotonic-relationships.png">

</div>
</div>

<!--s-->

## Correlation | Spearman Rank Correlation Coefficient

Spearman Rank Correlation Coefficient counts the number of disordered pairs, not how well the data fits a line. Thus, it is better for non-linear relationships. You can use the formula below only if all n ranks are distinct integers.

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

<div>
$ r_s = 1 - \frac{6 \sum_i d_i^2}{n^3 - n} $
</div>
<div>
$ d_i = \text{rank}(x_i) - \text{rank}(y_i) $
</div>


<span class="code-span">r_s = 1</span>: Perfect positive relationship <br>
<span class="code-span">r_s = -1</span>: Perfect negative relationship <br>
<span class="code-span">r_s = 0</span>: No relationship

</div>
<div class="c2" style = "width: 50%">

<img src="https://datatab.net/assets/tutorial/spearman/Calculate_Spearman_rank_correlation.png">
<p style="text-align: center; font-size: 0.6em; color: grey;"> Source: Datatab</p>

</div>
</div>


<!--s-->

## Correlation | Coefficients vs. Regression

- Correlation coefficients measure the strength of a linear relationship between two variables.
- Regression analysis can be used to describe the relationship between a dependent variable and one or more independent variables.

We'll cover regression in a later lecture, but it is relevant to understand the difference between the two.


<!--s-->

<div class="header-slide">

# Simpson's Paradox

</div>

<!--s-->

## Simpson's Paradox

Simpson's Paradox is a phenomenon in probability and statistics, in which a trend appears in different groups of data but **disappears** or **reverses** when these groups are combined.

Recall that correlation measures the strength of a linear relationship between two variables. But, always remember that correlation does not imply causation! 

**Simpson's Paradox** is a situation where a relationship is reversed when the data is split into subgroups.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*8tP_5zRKNAyVSeexu7RJZg.png" style = "border-radius: 15px; opacity: 0.6;">

<!--s-->

## Simpson's Paradox | Recovery Rate Example

Which hospital would you rather have surgery in? A or B?

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 16 | 784 | 2% |
| B | 63 | 2037 | 3% |

<!--s-->

## Simpson's Paradox | Recovery Rate Example

If you are in good condition, which hospital would you rather have surgery in? A or B?

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 8 | 592 | 1.3% |
| B | 6 | 594 | 1% |

<!--s-->

## Simpson's Paradox | Recovery Rate Example

If you are in poor condition, which hospital would you rather have surgery in? A or B?

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 8 | 192 | 4% |
| B | 57 | 1443 | 3.8% |

<!--s-->

## Simpson's Paradox | Recovery Rate Example

Let's look at all of the data together. Hospital B has a higher death rate than Hospital A in aggregate. But, when we look at the subgroups, Hospital A has a higher death rate in both subgroups.

<div style="font-size: 0.8em;">

### Overall

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 16 | 784 | 2% |
| B | 63 | 2037 | **3%** |

### Good Condition

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 8 | 592 | **1.3%** |
| B | 6 | 594 | 1% |

### Poor Condition

| Hospital | Died | Survived | Death Rate |
| --- | --- | --- | --- |
| A | 8 | 192 | **4%** |
| B | 57 | 1443 | 3.8% |

</div>

<!--s-->

## Simpson's Paradox | Linear Regression Example

Simpson's Paradox can also occur in linear regression or correlation analysis.

<img src="https://miro.medium.com/v2/resize:fit:1400/1*8tP_5zRKNAyVSeexu7RJZg.png" style = "border-radius: 15px; opacity: 0.6;">

<!--s-->

## Simpson's Paradox | What Is (Typically) Happening?

1. **Confounding Variables**: The relationship between the variables is influenced by a third variable.
2. **Sample Size**: The sample size of the subgroups is not large enough to capture the true relationship.

<!--s-->

## Simpson's Paradox | Prevention

1. **Segment Data Carefully**: Understand the context and how data groups are formed.
2. **Identify Confounders**: Look for variables that might be influencing the results.
3. **Holistic Approach**: Consider both combined and segmented data analyses.
4. **Use Visualizations**: Visualizations can help identify patterns and trends.

<!--s-->

<div class="header-slide">

# Association Analysis

</div>

<!--s-->

## Association Analysis | Definition

Association analysis measures the strength of co-occurrence between one item and another. It is widely applied in retail analysis of transactions, recommendation engines, online clickstream analysis, and more.

<!--s-->

## Association Analysis | Explanation

Given a set of transactions, association analysis finds rules that will predict the occurrence of an item based on the occurrences of other items.

For example, if a customer buys a product, what other products are they likely to buy?

<!--s-->

## Association Analysis | Definitions

- **Itemset**: A collection of one or more items
  - Example: <span class="code-span">{Milk, Bread, Diaper}</span>
  - **k-itemset**: An itemset that contains <span class="code-span">k</span> items
- **Association Rule**: An implication expression of the form <span class="code-span">X --> Y</span>, where <span class="code-span">X</span> and <span class="code-span">Y</span> are itemsets.
  - Example: <span class="code-span">{Milk, Diaper} --> {Beer}</span>

<!--s-->

## Association Rule | Evaluation

For a rule $ X \rightarrow Y $, where $ X \cap Y = \emptyset $, we can evaluate the rule using the following measures:

- **Support (S)**: Fraction of transactions that contain both X and Y. Where $\(T\)$ is the total number of transactions and $\sigma(X, Y)$ is the number of transactions that contain both $\(X\)$ and $\(Y\)$.

<div class="col-centered" style = "padding: 0.5em;">
$ S(X \rightarrow Y) = \frac{\sigma(X, Y)}{|T|} $
</div>

- **Confidence (C)**: Measures how often items in Y appear in transactions that contain X.

<div class="col-centered" style = "padding: 0.5em;">
$ C(X \rightarrow Y) = \frac{\text{S}(X, Y)}{\text{S}(X)} $
</div>

- **Lift (L)**: Takes into account the frequency of Y besides the confidence.

<div class="col-centered" style = "padding: 0.5em;">
$L(X \rightarrow Y) = \frac{S(X, Y)}{S(X)S(Y)}$
</div>

<!--s-->

## Association Rule | Example

Consider the following transactions:

| TID | Items |
| --- | --- |
| 1 | Bread, Milk |
| 2 | Bread, Diaper, Beer, Eggs |
| 3 | Milk, Diaper, Beer, Coke |
| 4 | Bread, Milk, Diaper, Beer |
| 5 | Bread, Milk, Diaper, Coke |

And the following association rule: <span class="code-span">{Milk, Diaper} --> {Beer}</span>

$ S = \frac{\sigma{\text(Milk, Diaper, Beer)}}{|T|} = \frac{2}{5} = 0.4 $

$ C = \frac{S(Milk, Diaper, Beer)}{S(Milk, Diaper)} = \frac{0.4}{0.6} = 0.67$

$ L = \frac{S(Milk, Diaper, Beer)}{S(Milk, Diaper)S(Beer)} = \frac{0.4}{0.6*0.6} = 1.11$


<!--s-->

## Association Analysis Rule Generation

Given a set of transactions \(T\), the goal of association rule mining is to find all rules having:

- Support $ \geq $ Support threshold
- Confidence $ \geq $ Confidence threshold

The goal is to find all rules that satisfy these constraints.

<!--s-->

## Association Analysis Rule Generation | Brute-force Approach

In order to get all of the possible association rules, we would need to:

  - List all possible association rules
  - Compute the support and confidence for each rule
  - Prune rules that fail the support or confidence thresholds

But, as with many ideal or simple solutions, this is computationally prohibitive.

<!--s-->

## Association Analysis Rule Generation | Apriori Principle

The Apriori principle is a fundamental concept in association rule mining. It states that if an itemset is frequent, then all of its subsets must also be frequent. 

This principle allows us to reduce the number of itemsets we need to consider when generating association rules, and thus reduce the computational complexity. Modern software implementations will use the Apriori principle to generate association rules.

<div class="col-centered">
<img src="https://www.researchgate.net/profile/Darshan-Tank/publication/276231946/figure/fig2/AS:406644845498372@1473963094797/An-illustration-of-the-Apriori-principle.png" style="border-radius:15px; height: 20%; width: 50%;">
</div>
<p style="text-align: center; font-size: 0.6em; color: grey;">Tank, Darshan. (2014)</p>

<!--s-->

## Summary

- **Correlation** measures the strength of a relationship between two variables.
- **Simpson's Paradox** is a phenomenon where a trend appears in different groups of data but disappears or reverses when these groups are combined.
- **Association Analysis** measures the strength of co-occurrence between one item and another.

<!--s-->

<div class = "header-slide">

# P.01 | Project Proposal
Due: 10.08.2024 <br>

</div>