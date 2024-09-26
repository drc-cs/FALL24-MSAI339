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

<div class = "header-slide">

# L.02 Data Sources

</div>

<!--s-->

## Data Sources

Data is at the heart of all data science. It's the raw material that we use to build models, make predictions, and draw conclusions. Data can be gathered from a variety of sources, and it comes in many different forms. 

<div class = "col-wrapper">
<div class = "c1" style = "width: 50%;">

Common Data Sources:

- Bulk Downloads
- APIs
- Scraping, Web Crawling
- BYOD (Bring Your Own Data)

</div>

<div class = "c2" style = "width: 50%">

<img src="https://imageio.forbes.com/blogs-images/gilpress/files/2016/03/Time-1200x511.jpg?format=jpg&width=1440" width="100%">
<p style="text-align: center; font-size: 0.6em; color: grey;">Forbes 2022</p>

</div>

<!--s-->

## Data Sources | Bulk Downloads

Large datasets that are available for download from the internet.


| Source | Examples | Advantages | Disadvantages |
| --- | --- | --- | --- |
| Government | data.gov | often free, very large datasets | Non-specific |
| Academic | UCI Machine Learning Repository | usually free, large datasets | Non-specific |
| Industry | Kaggle, HuggingFace | sometimes free, large datasets | Non-specific, sometimes expensive |


```python
# Demonstration of pulling from the UCI machine learning repository.
import pandas as pd
url = 'https://archive.ics.uci.edu/ml/machine-learning-databases/iris/iris.data'
df = pd.read_csv(url, header=None)
```

```python
# Demonstration of pulling from the HuggingFace datasets library.
from datasets import load_dataset
dataset = load_dataset('imdb')
```

<!--s-->

## Data Sources | APIs

Data APIs are often readily available for sign up (usually for a fee).

While offering a lot of data, APIs can be restrictive in terms of the data they provide (and the rate at which you can pull it!).

API's often have better structure (usually backed by a db), and they often have the additional benefit of streaming or real-time data that you can poll for updates.

```python
import requests

API_KEY = 'your_api_key'
url = f'https://api.yourapi.com/data?api_key={API_KEY}'
r = requests.get(url)
data = r.json()
```

```python
from openai import OpenAI
client = OpenAI()

response = client.chat.completions.create(
  model="gpt-3.5-turbo-0125",
  response_format={ "type": "json_object" },
  messages=[
    {"role": "system", "content": "You are a helpful assistant designed to output JSON."},
    {"role": "user", "content": "Who won the world series in 2020?"}
  ]
)
```

<!--s-->

## Data Sources | Scraping, Web Crawling

Web crawling is a free way to collect data from the internet. But be **cautious**. Many websites have terms of service that prohibit scraping, and you can easily overstep those.

Examples of packages built for webscraping in Python include <span class="code-span">beautifulsoup</span> and <span class="code-span">scrapy</span>.

```python
import requests
from bs4 import BeautifulSoup
url = "https://en.wikipedia.org/wiki/Python_(programming_language)"
r = requests.get(url)
soup = BeautifulSoup(r.text, 'html.parser')
soup.find_all('p')
```

<!--s-->

## Data Sources | BYOD (Bring Your Own Data)

Data that you collect yourself. In this case, you determine the format.

Examples for collecting survey data include Qualtrics and SurveyMonkey.

```python
import pandas as pd
df = pd.read_csv('survey_data.csv')
```
<!--s-->

## Data Sources | BYOD (Bring Your Own Data)
Collecting data yourself is common in academia and industry. But be careful!

<div class = "col-wrapper">
<div class = "c1" style = "width: 70%;">

**Policies Exist**
- *GDPR*: Controllers of personal data must put in place appropriate technical and organizational measures to implement the data protection principles.
- *HIPAA*: Covered entities must have in place appropriate administrative, technical, and physical safeguards to protect the privacy of protected health information.

**Bias *Always* Exists** Bias is everywhere. It's in the data you collect, the data you don't collect, and the data you use to train your models. In almost all cases, we perform a *sampling*. It's your job to ensure it is a reasonable sample.

**Ethical Concerns**
Don't collect data that you don't need. Definitely don't collect data that you don't have permission to collect.

</div>
<div class = "c2 col-centered"  style = "width: 30%">
<iframe width="256" height="256" src="https://www.youtube.com/embed/R2vhjC5qCQk?si=m3ImvZqhDawRoMTo" title="YouTube video player" frameborder="0" allow="accelerometer; autoplay; clipboard-write; encrypted-media; gyroscope; picture-in-picture; web-share" allowfullscreen></iframe>
</div>
</div>

<!--s-->

<div class = "header-slide">

# Data Structures

</div>
<!--s-->


## Data Structures

| Type | Example | Advantages | Disadvantages |
| --- | --- | --- | --- |
| Structured | Relational Database | easy to query, fast | not flexible, hard to update schema |
| Semi-Structured | XML, CSV, JSON | moderate flexibility, easy to add more data | slow, harder to query |
| Unstructured | Plain Text, Images, Audio, Video | very flexible, easy to add more data | slow, hardest to query |

<!--s-->

## Data Structures | Structured Example (Relational Database)

| Type | Example | Advantages | Disadvantages |
| --- | --- | --- | --- |
| Structured | Relational Database | easy to query, fast | not flexible, hard to update schema |
| Semi-Structured | XML, CSV, JSON | moderate flexibility, easy to add more data | slow, harder to query |
| Unstructured | Plain Text, Images, Audio, Video | very flexible, easy to add more data | slow, hardest to query |

```sql
CREATE TABLE employees (
  employee_id INT PRIMARY KEY,
  name VARCHAR(100),
  hire_date DATE
);
```
<!--s-->

## Data Structures | Semi-Structured Example (JSON)

| Type | Example | Advantages | Disadvantages |
| --- | --- | --- | --- |
| Structured | Relational Database | easy to query, fast | not flexible, hard to update schema |
| Semi-Structured | XML, CSV, JSON | moderate flexibility, easy to add more data | slow, harder to query |
| Unstructured | Plain Text, Images, Audio, Video | very flexible, easy to add more data | slow, hardest to query |

```json
employeeData = {
  "employee_id": 1234567,
  "name": "Jeff Fox",
  "hire_date": "1/1/2013"
};
```

<!--s-->

## Data Structures | Unstructured Example (Plain Text)

| Type | Example | Advantages | Disadvantages |
| --- | --- | --- | --- |
| Structured | Relational Database | easy to query, fast | not flexible, hard to update schema |
| Semi-Structured | XML, CSV, JSON | moderate flexibility, easy to add more data | slow, harder to query |
| Unstructured | Plain Text, Images, Audio, Video | very flexible, easy to add more data | slow, hardest to query |

```text
Dear Sir or Madam,

My name is Jeff Fox (employee id 1234567). I'm excited to start on 1/1/2013.

Sincerely,
Jeff Fox
```

<!--s-->

<div class = "header-slide">

# SQL Overview

</div>

<!--s-->

### SQL Query Cheat Sheet (Part 1)

### `CREATE TABLE`

  ```sql
  /* Create a table called table_name with column1, column2, and column3. */
  CREATE TABLE table_name (
    column1 INT PRIMARY KEY, /* Primary key is a unique identifier for each row. */
    column2 VARCHAR(100), /* VARCHAR is a variable-length string up to 100 characters. */
    column3 DATE /* DATE is a date type. */
  );
  ```

### `INSERT INTO`

  ```sql
  /* Insert values into column1, column2, and column3 in table_name. */
  INSERT INTO table_name (column1, column2, column3) VALUES (value1, value2, value3);
  ```

### `UPDATE`

  ```sql
  /* Update column1 in table_name to 'value' where column2 is equal to 'value'. */
  UPDATE table_name SET column1 = 'value' WHERE column2 = 'value';
  ```

### `DELETE`

  ```sql
  /* Delete from table_name where column1 is equal to 'value'. */
  DELETE FROM table_name WHERE column1 = 'value';
  ```

<!--s-->

### SQL Query Cheat Sheet (Part 2)

### `SELECT`
  
  ```sql
  /* Select column1 and column2 from table_name.*/
  SELECT column1, column2 FROM table_name;
  ```

### `WHERE`

  ```sql
  /* Select column1 and column2 from table_name where column1 is equal to 'value' and column2 is equal to 'value'. */
  SELECT column1, column2 FROM table_name WHERE column1 = 'value' AND column2 = 'value';
  ```

### `ORDER BY`

  ```sql
  /* Select column1 and column2 from table_name and order by column1 in descending order. */
  SELECT column1, column2 FROM table_name ORDER BY column1 DESC;
  ```

### `LIMIT`

  ```sql
  /* Select column1 and column2 from table_name and limit the results to 10. */
  SELECT column1, column2 FROM table_name LIMIT 10;
  ```
<!--s-->

### SQL Query Cheat Sheet (Part 3)

### `JOIN`

  ```sql
  /* Select column1 and column2 from table1 and table2 where column1 is equal to column2. */
  SELECT column1, column2 FROM table1 JOIN table2 ON table1.column1 = table2.column2;
  ```

### `GROUP BY`

  ```sql
  /* Select column1 and column2 from table_name and group by column1. */
  SELECT column1, column2 FROM table_name GROUP BY column1;
  ```

### `COUNT`

  ```sql
  /* Select the count of column1 from table_name. */
  SELECT COUNT(column1) FROM table_name;

  /* Group by column2 and select the count of column1 from table_name. */
  SELECT column2, COUNT(column1) FROM table_name GROUP BY column2;
  ```

### `SUM`

  ```sql
  /* Select the sum of column1 from table_name. */
  SELECT SUM(column1) FROM table_name;

  /* Group by column2 and select the sum of column1 from table_name. */
  SELECT column2, SUM(column1) FROM table_name GROUP BY column2;
  ```

<!--s-->

### SQL and Pandas (🔥)

```python
import pandas as pd
import sqlite3

# Create a connection to a SQLite database.
conn = sqlite3.connect('example.db')

# Load a DataFrame into the database.
df = pd.DataFrame({'column1': [1, 2, 3], 'column2': ['a', 'b', 'c']})
df.to_sql('table_name', conn, if_exists='replace')

# Query the database.
query = 'SELECT * FROM table_name'
df = pd.read_sql(query, conn)
```

<!--s-->

<div class = "col-wrapper">
  <div class="c1 col-centered">
  <div style="font-size: 0.8em; left: 0; width: 50%; position: absolute;">

  ## SQL Challenge | Q.01

  
  Write a SQL query that selects the <span class = "code-span">name</span> and <span class = "code-span">year</span> columns from a <span class = "code-span">movies</span> table where the <span class = "code-span">year</span> is greater than 2000.

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

  ## SQL Challenge | Q.02

  Write a SQL query that selects the <span class = "code-span">name</span> and <span class = "code-span">year</span> columns from a <span class = "code-span">movies</span> table where the <span class = "code-span">year</span> is greater than 2000 and the <span class = "code-span">genre</span> is equal to <span class = "code-span">comedy</span>.

  </div>
  </div>
  <div class="c2 col-centered" style = "bottom: 0; right: 0; width: 100%; padding-top: 5%">
  <img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="50%">
  </div>
</div>

<!--s-->

## Practice Your SQL!

The majority of data science interviews will have a SQL component. It's a good idea to practice your SQL skills. Here are a few resources to get you started:

- ### [SQLZoo](https://sqlzoo.net/)
- ### [W3Schools](https://www.w3schools.com/sql/)
- ### [LeetCode](https://leetcode.com/problemset/database/)
- ### [HackerRank](https://www.hackerrank.com/domains/sql)
- ### [SQL Practice](https://www.sql-practice.com/)

<!--s-->

<div class = "header-slide">

# Data Cleaning

</div>
<!--s-->

## Data Cleaning

Data is often dirty! Don't ever give your machine learning or statistical model dirty data.

Remember the age-old adage:

> Garbage in, garbage out.

 Data cleaning is the process of converting source data into target data without errors, duplicates, or inconsistencies. You will often need to structure data in a way that is useful for your analysis, so learning some basic data manipulation is **essential**.

<!--s-->

## Data Cleaning | Common Data Issues

1. Incompatible data
2. Missing values
3. Extreme Outliers

<!--s-->

## Data Cleaning | Handling Incompatible Data

<div style = "font-size: 0.85em;">

| Data Issue | Description | Example | Solution |
| --- | --- | --- | --- |
| Unit Conversions | Numerical data conversions can be tricky. | 1 mile != 1.6 km | Measure in a common unit, or convert with caution. |
| Precision Representations | Data can be represented differently in different programs. | 64-bit float to 16-bit integer | Use the precision necessary and hold consistent. |
| Character Representations | Data is in different character encodings. | ASCII, UTF-8, ... | Create using the same encoding, or convert with caution. |
| Text Unification | Data is in different formats. | D'Arcy; Darcy; DArcy; D Arcy; D&-#-3-9-;Arcy | Use a common format, or convert with caution. <span class="code-span">RegEx</span> will be your best friend.| 
| Time / Date Unification | Data is in different formats. | 10/11/2019 vs 11/10/2019 | Use standard libraries & UTC. A personal favorite is seconds since epoch. |

</div>
<!--s-->

## Data Cleaning | Handling Missing Values

<div class = "col-wrapper">
<div class="c1" style = "width: 50%">

  Data is often missing from datasets. It's important to identify why it's missing. Once you have established that it is **missing at random**, you can proceed with **substitution**.

  When missing data, we have a few options at our disposal:

  1. Drop the entire row
  2. Drop the entire column
  3. Substitute with a reasonable value

</div>
<div class="c2" style = "width: 50%">

  ``` [1-5|4]
  id    col1    col2    col3    col4
  p1    2da6    0.99    32     43
  p2    cd2d    1.23    55      38
  p3    9999    89.2    NaN     32
  p4    4e7c    0.72    9.7     35
  ```

</div>
</div>

<!--s-->

## Data Cleaning | Handling Missing Values with Substitution

<div class = "col-wrapper"> 
<div class = "c1" style = "width: 70%; font-size: 0.85em;">


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

<div class = "c2 col-centered" style = "width: 40%">

```python
import pandas as pd

# Load data.
df = pd.read_csv('data.csv')

# Forward fill.
df_ffill = df.fillna(method='ffill')

# Backward fill.
df_bfill = df.fillna(method='bfill')

# Mean value imputation.
df_mean = df.fillna(df.mean())

# Random value imputation.
df_random = df.fillna(df.sample())

# Imputation by interpolation.
df_interpolate = df.interpolate()
```

</div>
</div>

<!--s-->

## Data Cleaning | Q.03

You have streaming data that is occasionally dropping values. Which of the following methods would be appropriate to fill missing values when signal fails to update? 

*Please note, in this scenario, you can't use the future to predict the past.*

<div class="col-wrapper">
<div class="c1 col-centered" style = "width: 60%; padding-bottom: 20%;">

<div style = "line-height: 2em;">
&emsp;A. Forward fill <br>
&emsp;B. Imputation by interpolation <br>
&emsp;C. Mean value imputation <br>
&emsp;D. Backward fill <br>
</div>

</div>

<div class="c2 col-centered" style = "width: 40%; padding-bottom: 20%">
<img src="https://storage.googleapis.com/slide_assets/PollEverywhere.png" width="100%">
<a>poll.ev.com/nucs</a>
</div>
</div>

<!--s-->

## Data Cleaning | Handling Outliers

Outliers are extreme values that deviate from other observations on data. They may indicate a variability in a measurement, experimental errors, or a novelty.

Outliers can be detected using:

- **Boxplots**: A visual representation of the five-number summary.
- **Scatterplots**: A visual representation of the relationship between two variables.
- **z-scores**: A measure of how many standard deviations a data point is from the mean.
- **IQR**: A measure of statistical dispersion, being equal to the difference between the upper and lower quartiles.

Handling outliers should be done on a case-by-case basis. Don't throw away data unless you have a very compelling (and **documented**) reason to do so!

<!--s-->

## Data Cleaning | Keep in Mind

- Cleaning your data is an iterative process.
  - **Hot tip**: 🔥 Focus on making your data preprocessing *fast*. You will be doing it a lot, and you'll want to be able to iterate quickly. Look into libraries like <span class="code-span">dask</span>, <span class="code-span">pyspark</span>, and <span class="code-span">ray</span> for large datasets.
  - **Hot tip**: 🔥 Platforms like DataFlow and BigQuery automate a lot of minutia and give you parallelization out-of-the-box.

- Data cleaning is often planned with visualization. 
  - Always look at the data. Always. We'll go over systematic EDA approaches in L.03.
- Data cleaning can fix modeling problems.
  - Your first assumption should always be "something is wrong with the data", not "I should try another model".
- Data cleaning is not a one-size-fits-all process, and often requires domain expertise.

<!--s-->

<div class = "header-slide">

# P.01 | Group Formation & Project Proposal
Due: 10.08.2024 <br>
Canvas Assignment Link: P.01

</div>

<!--s-->


## P.01 | Creating a Group

### **Group Creation**
Please form groups of **4**. You can create a group on Canvas using this [[link](https://canvas.northwestern.edu/courses/216743/groups)]. Click on the "Create Group" button OR "Join Group" button.

### **Group Name**
Your group name can be anything you like, but it should be unique.

### **Communication**
I suggest creating a group chat on Slack, Discord, or WhatsApp. This will make it easier to communicate with your group members.

<!--s-->

## P.01 | Project Proposal

**One person** per group should submit the project proposal using the text editor on Canvas. Please follow the format provided on canvas. **P.01 is due on 10.08.2024**.

Your proposal should include:

1. **Team Name**: The name of your team.
2. **Team Members**: The names of your team members.
3. **Project Title**: The title of your project.
4. **Project Description**: A brief description of your project.
5. **Data Source**: Where you plan to get your data.
6. **Project Goals**:
    - What are you trying to accomplish?
    - What are the deliverables? Are you creating an app, a report, a model, etc.?

<!--s-->

<div class="header-slide">

# Questions?

</div>