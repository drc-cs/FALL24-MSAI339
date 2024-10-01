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
  ## L.02 | Data Sources

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

- **H.01** is due 10.01.2024 at 11:59 PM.
    - 100% of you have submitted 🎉.

- **P.01** is due 10.08.2024 at 11:59 PM.

- **GA Office Hours**:
    - Wednesdays from 1-2PM in Annenberg G28.

- **Office Hours**:
    - Send me an email -- I'm on campus Tuesdays and Thursdays. 

<!--s-->

## Exploratory Data Analysis (EDA) | Introduction

**Exploratory Data Analysis** (EDA) is an approach to analyzing data sets to summarize their main characteristics, often with visual methods.

This should be done every single time you are exposed to a new dataset. **ALWAYS** look at the data (x3).

EDA will help you to identify early issues or patterns in the data, and will guide you in the next steps of your analysis. It is an absolutely **critical**, but often overlooked step.

<!--s-->

## Exploratory Data Analysis (EDA) | Methodology

We can break down EDA into two main categories:

- **Descriptive EDA**: Summarizing the main characteristics of the data.
- **Graphical EDA**: Visualizing the data to understand its structure and patterns.

<!--s-->

<div class="header-slide"> 

# Descriptive EDA

</div>

<!--s-->

## Descriptive EDA | Examples

- **Central tendency**
    - Mean, Median, Mode
- **Spread**
    - Range, Variance, interquartile range (IQR)
- **Skewness**
    - A measure of the asymmetry of the distribution. Typically close to 0 for a normal distribution.
- **Kurtosis**
    - A measure of the "tailedness" of the distribution. Typically close to 3 for a normal distribution.
- **Modality**
    - The number of peaks in the distribution.

<!--s-->

## Central Tendency

- **Mean**: The average of the data. 

    - $ \bar{x} = \frac{1}{n} \sum_{i=1}^{n} x_i $

    - <span class="code-span">np.mean(data)</span>

- **Median**: The middle value of the data, when sorted.

    - [1, 2, **4**, 5, 6]

    - <span class="code-span">np.median(data)</span>

- **Mode**: The most frequent value in the data.

    ```python
    from scipy.stats import mode
    data = np.random.normal(0, 1, 1000)
    mode(data)
    ```

<!--s-->

## Spread

- **Range**: The difference between the maximum and minimum values in the data.
    
    - <span class="code-span">np.max(data) - np.min(data)</span>

- **Variance**: The average of the squared differences from the mean.

    - $ \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 $

    - <span class="code-span">np.var(data)</span>

- **Standard Deviation**: The square root of the variance.

    - $ \sqrt{\frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2} $
    - <span class="code-span">np.std(data)</span>

- **Interquartile Range (IQR)**: The difference between the 75th and 25th percentiles.
    - <span class="code-span">np.percentile(data, 75) - np.percentile(data, 25)</span>

<!--s-->

## Skewness

A measure of the lack of "symmetry" in the data.

**Positive skew (> 0)**: the right tail is longer; the mass of the distribution is concentrated on the left of the figure.

**Negative skew (< 0)**: the left tail is longer; the mass of the distribution is concentrated on the right of the figure.

```python
import numpy as np
from scipy.stats import skew

data = np.random.normal(0, 1, 1000)
print(skew(data))
```

<!--s-->

## Skewness | Plot

<iframe width = "80%" height = "80%" src="https://storage.googleapis.com/cs326-bucket/lecture_3_plots/skewness.html" title="scatter_plot"></iframe>

<!--s-->

## Kurtosis

A measure of the "tailedness" of the distribution.

- **Leptokurtic (> 3)**: the tails are fatter than the normal distribution.

- **Mesokurtic (3)**: the tails are the same as the normal distribution.

- **Platykurtic (< 3)**: the tails are thinner than the normal distribution.

```python
import numpy as np
from scipy.stats import kurtosis

data = np.random.normal(0, 1, 1000)
print(kurtosis(data))
```

<!--s-->

## Kurtosis | Plot

<img width = "100%" height = "100%" src="https://www.scribbr.com/wp-content/uploads/2022/07/The-difference-between-skewness-and-kurtosis.webp" title="scatter_plot"></img>

<!--s-->

## Moments

In statistics, the **moments** of a distribution are a set of quantities that describe its shape.

| Moment | Description | Simple Formula | Information |
| --- | --- | --- | --- |
| 1 | Mean | $ \frac{1}{n} \sum_{i=1}^{n} x_i $ |
| 2 | Variance | $ \frac{1}{n} \sum_{i=1}^{n} (x_i - \bar{x})^2 $ |
| 3 | Skewness | $ \frac{1}{n} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{\sigma}\right)^3 $ | [Wikipedia](https://en.wikipedia.org/wiki/Skewness)
| 4 | Kurtosis | $ \frac{1}{n} \sum_{i=1}^{n} \left(\frac{x_i - \bar{x}}{\sigma}\right)^4 $ | [Wikipedia](https://en.wikipedia.org/wiki/Kurtosis)|


<!--s-->

## Modality

<div class = "col-wrapper">

<div class="c1" style = "width: 50%; padding-top: 10%;">

The number of peaks in the distribution.

- **Unimodal**: One peak.
- **Bimodal**: Two peaks.
- **Multimodal**: More than two peaks.

</div>

<div class="c2 col-centered" style = "width: 50%;">

<img src="https://treesinspace.com/wp-content/uploads/2016/01/modes.png?w=584" width="100%">

</div>


<!--s-->

## Normal Distribution | Definition

A normal distribution is a continuous probability distribution that is symmetric about the mean, showing that data near the mean are more frequent in occurrence than data far from the mean. Mathematically, it is defined as:

<div style = "text-align: center;">
$ f(x | \mu, \sigma^2) = \frac{1}{\sqrt{2\pi\sigma^2}} e^{-\frac{(x - \mu)^2}{2\sigma^2}} $
</div>

Where:

- $ \mu $ is the mean.
- $ \sigma^2 $ is the variance.

<!--s-->
## Normal Distribution | Properties

The normal distribution has several key properties:

- **Symmetry**: The distribution is symmetric about the mean.
- **Unimodality**: The distribution has a single peak.
- **68-95-99.7 Rule**: 
    - 68% of the data falls within 1 standard deviation of the mean.
    - 95% within 2 standard deviations.
    - 99.7% within 3 standard deviations.

<!--s-->

## Normal Distribution | Properties

How can we tell if our data is normally distributed?


<div class = "col-wrapper">
<div class="c1" style = "width: 50%; padding-top: 10%;">

- **Skewness**: close to 0.
- **Kurtosis**: close to 3.
- **QQ Plot**: A plot of the quantiles of the data against the quantiles of the normal distribution.
- **Shapiro-Wilk Test**: A statistical test to determine if the data is normally distributed.

</div>
<div class="c2 col-centered" style = "width: 50%;">

<img src="https://miro.medium.com/v2/format:webp/0*lsbu809F5gOZrrKX" width="100%">

</div>

<!--s-->

## Descriptive EDA | Not Covered

There are many ways to summarize data, and we have only covered a few of them. Here are some common methods that we did not cover today:

- **Covariance**: A measure of the relationship between two variables (L.04).
- **Correlation**: A normalized measure of the relationship between two variables (L.04).
- **Outliers**: Data points that are significantly different from the rest of the data (L.02).
- **Missing Values**: Data points that are missing from the dataset (L.02).
- **Percentiles**: The value below which a given percentage of the data falls.
- **Frequency**: The number of times a value occurs in the data.

<!--s-->

<div class="header-slide">

# Graphical EDA 

</div>

<!--s-->

## Graphical EDA | Data Types

There are three primary types of data -- nominal, ordinal, and numerical.

| Data Type | Definition | Example |
| --- | --- | --- |
| Nominal | Categorical data without an inherent order | <span class="code-span">["red", "green", "orange"]</span> |
| Ordinal | Categorical data with an inherent order | <span class="code-span">["small", "medium", "large"]</span> <p><span class="code-span">["1", "2", "3"]</span> |
| Numerical | Continuous or discrete numerical data | <span class="code-span">[3.1, 2.1, 2.4]</span> |

<!--s-->

## Graphical EDA | Choosing a Visualization

The type of visualization you choose will depend on:

- **Data type**: nominal, ordinal, numerical.
- **Dimensionality**: 1D, 2D, 3D+.
- **Story**: The story you want to tell with the data.

Whatever type of plot you choose, make sure your visualization is information dense **and** easy to interpret. It should always be clear what the plot is trying to convey.

<!--s-->

## Graphical EDA | Common Visualization Types

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

### 1D Data
- Bar chart
- Pie chart
- Histogram
- Boxplot
- Violin plot
- Line plot

### 2D Data
- Scatter plot
- Heatmap
- Bubble plot
- Line plot
- Boxplot
- Violin plot

</div>
<div class="c2 col-centered" style = "width: 50%;">

### 3D+ Data

- 3D scatter plot
- Bubble plot
- Color scatter plot
- Scatter plot matrix

</div>
</div>

<!--s-->

## A Note on Tools

Matplotlib and Plotly are the most popular libraries for data visualization in Python.

| Library | Pros | Cons |
| --- | --- | --- |
| Matplotlib | Excellent for static publication-quality plots, very fast render, old and well supported. | Steeper learning curve, many ways to do the same thing, no interactivity, OOTB color schemes. |
| Plotly | Excellent for interactive plots, easy to use, easy tooling for animations, built-in support for dashboarding and publishing online. | Not as good for static plots, less fine-grained control, high density renders can be non-trivial. |

<!--s-->

## 1D Data | Histograms

When you have numerical data, histograms are a great way to visualize the distribution of the data. If there is a clear distribution, it's often useful to fit a probability density function (PDF).

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python
import numpy as np
from plotly import express as px

data = np.random.normal(0, 1, 100)
fig = px.histogram(data, nbins = 50)
fig.show()
```

</div>
<div class="c2 col-centered" style = "width: 60%; padding: 0; margin: 0;">

<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/cs326-bucket/lecture_3_plots/histogram.html" title="scatter_plot" padding=2em;></iframe>

</div>
</div>

<!--s-->


## 1D Data | Boxplots

Boxplots are a great way to visualize the distribution of the data, and to identify outliers.

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python
import numpy as np
from plotly import express as px

data = np.random.normal(0, 1, 100)
fig = px.box(data)
fig.show()
```

</div>
<div class="c2 col-centered" style = "width: 60%; padding: 0; margin: 0;">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/cs326-bucket/lecture_3_plots/boxplot.html" title="scatter_plot" padding=2em;></iframe>
</div>
</div>

<!--s-->

## 1D Data | Violin Plots

Violin plots are similar to box plots, but they also show the probability density of the data at different values.

<div class = "col-wrapper">

<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python

import numpy as np
from plotly import express as px

data = np.random.normal(0, 1, 100)
fig = px.violin(data)
fig.show()
```

</div>
<div class="c2 col-centered" style = "width: 60%; padding: 0; margin: 0;">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/cs326-bucket/lecture_3_plots/violin.html" title="scatter_plot" padding=2em;></iframe>
</div>
</div>

<!--s-->

## 1D Data | Bar Charts

Bar charts are a great way to visualize the distribution of **categorical** data.

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python

import numpy as np
from plotly import express as px

data = np.random.choice(["A", "B", "C"], 1000)
fig = px.histogram(data)
fig.show()
```

</div>
<div class="c2 col-centered" style = "width: 60%; padding: 0; margin: 0;">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/cs326-bucket/lecture_3_plots/bar.html" title="scatter_plot" padding=2em;></iframe>
</div>
</div>

<!--s-->

## 1D Data | Pie Charts

Pie charts are another way to visualize the distribution of categorical data.

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python

import numpy as np
from plotly import express as px

data = np.random.choice(["A", "B", "C"], 100)
fig = px.pie(data)
fig.show()
```

</div>
<div class="c2 col-centered" style = "width: 60%; padding: 0; margin: 0;">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/cs326-bucket/lecture_3_plots/pie.html" title="scatter_plot"  padding=2em;></iframe>
</div>
</div>

<!--s-->

## 2D Data | Scatter Plots

Scatter plots can help visualize the relationship between two numerical variables.

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python
import numpy as np
from plotly import express as px

x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
fig = px.scatter(x = x, y = y)
fig.show()
```

</div>
<div class="c2 col-centered" style = "width: 60%; padding: 0; margin: 0;">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/cs326-bucket/lecture_3_plots/scatter.html" title="scatter_plot"  padding=2em;></iframe>
</div>
</div>

<!--s-->

## 2D Data | Heatmaps (2D Histogram)

Heatmaps help to visualize the density of data in 2D.

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python
import numpy as np
from plotly import express as px

x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
fig = px.density_heatmap(x = x, y = y)
fig.show()
```

</div>
<div class="c2" style = "width: 60%">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/cs326-bucket/lecture_3_plots/heatmap.html" title="scatter_plot"  padding=2em;></iframe>
</div>
</div>

<!--s-->

## 3D+ Data | Bubble Plots

Bubble plots are a great way to visualize the relationship between three numerical variables and a categorical variable.


<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python
import numpy as np
from plotly import express as px

x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
z = np.random.normal(0, 1, 100)
c = np.random.choice(["A", "B", "C"], 100)
fig = px.scatter(x = x, y = y, size = z, color = c)
fig.show()
```

</div>
<div class="c2" style = "width: 60%">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/cs326-bucket/lecture_3_plots/bubble_plot.html" title="scatter_plot" padding=2em;></iframe>
</div>
</div>

<!--s-->

## 3D+ Data | Scatter Plots

Instead of using the size of the markers (as in the bubble plot), you can use another axis to represent a third numerical variable. And, you still have the option to color by a categorical variable.

<div class = "col-wrapper">
<div class="c1 col-centered" style = "width: 40%; padding: 0; margin: 0;">

```python
import numpy as np
from plotly import express as px

x = np.random.normal(0, 1, 100)
y = np.random.normal(0, 1, 100)
z = np.random.normal(0, 1, 100)
c = np.random.choice(["A", "B", "C"], 100)

fig = px.scatter_3d(x = x, y = y, z = z, color = c)
fig.show()
```

</div>
<div class="c2" style = "width: 50%">
<iframe width = "100%" height = "100%" src="https://storage.googleapis.com/cs326-bucket/lecture_3_plots/scatter_3d.html" title="scatter_plot" padding=2em;></iframe>
</div>
</div>

<!--s-->

## TLDR;

- **Descriptive EDA**: Summarizing the main characteristics of the data.
- **Graphical EDA**: Visualizing the data to understand its structure and patterns.

<!--s-->

<div class = "header-slide">

# H.01 | hello_world.py
Due: 10.01.2024 <br>

</div>

<!--s-->

<div class = "header-slide">

# P.01 | Project Proposal
Due: 10.08.2024 <br>

</div>