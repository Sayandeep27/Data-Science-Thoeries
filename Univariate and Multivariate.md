# 📊 Univariate, Bivariate, and Multivariate Analysis in Data Science

A complete, beginner‑friendly yet **industry‑ready guide** to understanding **Univariate, Bivariate, and Multivariate Analysis** with **theory, intuition, tables, visual explanations, and Python code examples**.

---

## 📌 Table of Contents

1. What is Data Analysis?
2. Why Do We Need Univariate, Bivariate & Multivariate Analysis?
3. Univariate Analysis

   * Definition
   * Types of Variables
   * Numerical Variable Analysis
   * Categorical Variable Analysis
   * Common Plots
   * Python Code Examples
4. Bivariate Analysis

   * Definition
   * Numerical vs Numerical
   * Categorical vs Numerical
   * Categorical vs Categorical
   * Correlation Concepts
   * Python Code Examples
5. Multivariate Analysis

   * Definition
   * Why Multivariate Analysis is Important
   * Common Techniques
   * Multivariate Visualization
   * Python Code Examples
6. Comparison Table
7. Real‑World Case Study Example
8. Key Takeaways

---

## 1️⃣ What is Data Analysis?

**Data Analysis** is the process of inspecting, cleaning, transforming, and modeling data to:

* Discover patterns
* Extract insights
* Support decision‑making

In **Exploratory Data Analysis (EDA)**, we mainly start with:

* Univariate Analysis
* Bivariate Analysis
* Multivariate Analysis

---

## 2️⃣ Why Do We Need These Analyses?

| Analysis Type | Purpose                                          |
| ------------- | ------------------------------------------------ |
| Univariate    | Understand individual variables                  |
| Bivariate     | Understand relationships between two variables   |
| Multivariate  | Understand interactions among multiple variables |

EDA always starts **simple → complex**.

---

## 3️⃣ Univariate Analysis

### 🔹 Definition

**Univariate Analysis** analyzes **one variable at a time**.

It answers:

* What values does this variable take?
* How is it distributed?
* Are there outliers?

---

### 🔹 Types of Variables

| Variable Type | Examples                 |
| ------------- | ------------------------ |
| Numerical     | Age, Salary, Height      |
| Categorical   | Gender, City, Department |

---

### 🔹 Univariate Analysis – Numerical Variables

#### Common Statistics

* Mean
* Median
* Mode
* Min / Max
* Variance
* Standard Deviation
* Quartiles

#### Example

```python
import pandas as pd
import numpy as np

age = pd.Series([22, 25, 30, 35, 40, 28, 26, 50])

print("Mean:", age.mean())
print("Median:", age.median())
print("Standard Deviation:", age.std())
print("Minimum:", age.min())
print("Maximum:", age.max())
```

---

### 🔹 Visualization – Numerical Variable

```python
import matplotlib.pyplot as plt

plt.hist(age, bins=5)
plt.xlabel("Age")
plt.ylabel("Frequency")
plt.title("Age Distribution")
plt.show()
```

---

### 🔹 Univariate Analysis – Categorical Variables

Focuses on:

* Frequency counts
* Proportions

#### Example

```python
gender = pd.Series(['Male', 'Female', 'Female', 'Male', 'Male', 'Female'])
print(gender.value_counts())
```

#### Visualization

```python
gender.value_counts().plot(kind='bar')
plt.xlabel("Gender")
plt.ylabel("Count")
plt.title("Gender Distribution")
plt.show()
```

---

## 4️⃣ Bivariate Analysis

### 🔹 Definition

**Bivariate Analysis** studies the **relationship between two variables**.

It answers:

* Are the variables related?
* How strong is the relationship?
* Is it positive or negative?

---

### 🔹 Types of Bivariate Analysis

| Variable 1  | Variable 2  | Method                     |
| ----------- | ----------- | -------------------------- |
| Numerical   | Numerical   | Correlation, Scatter Plot  |
| Numerical   | Categorical | Box Plot, Group Statistics |
| Categorical | Categorical | Crosstab, Chi‑Square       |

---

### 🔹 Numerical vs Numerical

#### Example: Height vs Weight

```python
height = [150, 160, 165, 170, 175, 180]
weight = [50, 55, 60, 65, 70, 80]

plt.scatter(height, weight)
plt.xlabel("Height")
plt.ylabel("Weight")
plt.title("Height vs Weight")
plt.show()
```

#### Correlation

```python
import numpy as np
print(np.corrcoef(height, weight))
```

---

### 🔹 Numerical vs Categorical

#### Example: Salary by Department

```python
data = pd.DataFrame({
    'Department': ['IT', 'HR', 'IT', 'Finance', 'HR', 'Finance'],
    'Salary': [60000, 45000, 65000, 70000, 48000, 72000]
})

import seaborn as sns
sns.boxplot(x='Department', y='Salary', data=data)
plt.show()
```

---

### 🔹 Categorical vs Categorical

#### Example

```python
pd.crosstab(data['Department'], ['High Salary' if s > 60000 else 'Low Salary' for s in data['Salary']])
```

---

## 5️⃣ Multivariate Analysis

### 🔹 Definition

**Multivariate Analysis** analyzes **more than two variables simultaneously**.

It helps understand:

* Combined effects
* Complex relationships
* Feature interactions

---

### 🔹 Why Multivariate Analysis Matters

Real‑world decisions **never depend on one variable**.

Example:

> Loan approval depends on income, credit score, age, job type, and debt.

---

### 🔹 Common Multivariate Techniques

| Technique           | Use Case                         |
| ------------------- | -------------------------------- |
| Pair Plot           | Multiple bivariate relationships |
| Correlation Matrix  | Strength of relationships        |
| PCA                 | Dimensionality Reduction         |
| Multiple Regression | Predict outcome                  |
| Clustering          | Customer segmentation            |

---

### 🔹 Pair Plot Example

```python
sns.pairplot(data[['Salary']])
plt.show()
```

---

### 🔹 Correlation Matrix

```python
corr = data[['Salary']].corr()
sns.heatmap(corr, annot=True)
plt.show()
```

---

### 🔹 Multivariate Regression Example

```python
from sklearn.linear_model import LinearRegression

X = [[25, 50000], [30, 60000], [35, 70000]]  # Age, Income
y = [200, 250, 300]  # Spending Score

model = LinearRegression()
model.fit(X, y)

print(model.coef_)
print(model.intercept_)
```

---

## 6️⃣ Comparison Table

| Feature    | Univariate   | Bivariate    | Multivariate |
| ---------- | ------------ | ------------ | ------------ |
| Variables  | 1            | 2            | 3+           |
| Focus      | Distribution | Relationship | Interaction  |
| Complexity | Low          | Medium       | High         |
| Examples   | Histogram    | Scatter Plot | Regression   |

---

## 7️⃣ Real‑World Case Study

### 🎯 Problem: Customer Churn Analysis

| Step   | Analysis Type | Purpose                |
| ------ | ------------- | ---------------------- |
| Step 1 | Univariate    | Understand churn rate  |
| Step 2 | Bivariate     | Churn vs Salary        |
| Step 3 | Multivariate  | Predict churn using ML |

---

## 8️⃣ Key Takeaways

* Always start with **Univariate Analysis**
* Use **Bivariate Analysis** to validate assumptions
* Use **Multivariate Analysis** for real‑world modeling
* Visualization is as important as statistics
* EDA builds the foundation for Machine Learning

---

## ✅ Final Note

This README is **GitHub‑ready**, **interview‑oriented**, and **project‑friendly**.
You can directly:

* Download it
* Add it to your portfolio
* Use it for revision before interviews

---

Happy Learning 🚀
