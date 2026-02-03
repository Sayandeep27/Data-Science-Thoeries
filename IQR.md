# 📦 Interquartile Range (IQR) for Outlier Detection

A **complete, beginner-to-advanced, GitHub-ready guide** explaining **IQR (Interquartile Range)** for **outlier detection** with intuition, theory, visuals (described), examples, edge cases, and **Python code**.

---

## 📌 Table of Contents

1. What Are Outliers?
2. Why Outlier Detection Matters
3. What Is IQR?
4. Quartiles Explained (Q1, Q2, Q3)
5. Mathematical Definition of IQR
6. How IQR Detects Outliers (Logic)
7. Step-by-Step IQR Calculation (Manual Example)
8. Visual Interpretation (Boxplot Logic)
9. IQR Outlier Detection Formula
10. Python Implementation (From Scratch)
11. IQR Using NumPy & Pandas
12. IQR with Real Dataset Example
13. Handling Outliers (Remove, Cap, Transform)
14. IQR vs Z-Score (Comparison Table)
15. When to Use IQR (Best Practices)
16. Common Mistakes & Edge Cases
17. IQR in Machine Learning Pipelines
18. Interview Questions
19. Summary Cheat Sheet

---

## 1️⃣ What Are Outliers?

**Outliers** are data points that are **very different** from most other values in a dataset.

### Example

```
Salary (₹): [25k, 30k, 28k, 27k, 26k, 2,50,000]
```

👉 `2,50,000` is clearly an **outlier**.

---

## 2️⃣ Why Outlier Detection Matters

Outliers can:

* Skew **mean & standard deviation**
* Break **linear regression assumptions**
* Reduce **model accuracy**
* Cause **overfitting**

✅ Hence, detecting and treating outliers is **critical** in data science.

---

## 3️⃣ What Is IQR?

**IQR (Interquartile Range)** measures the **spread of the middle 50% of data**.

📌 It focuses on **robust statistics** (not affected by extreme values).

> IQR is one of the **most reliable methods** for outlier detection.

---

## 4️⃣ Quartiles Explained

| Quartile | Meaning                          |
| -------- | -------------------------------- |
| Q1       | 25th percentile (lower quartile) |
| Q2       | 50th percentile (median)         |
| Q3       | 75th percentile (upper quartile) |

📌 Quartiles divide data into **4 equal parts**.

---

## 5️⃣ Mathematical Definition of IQR

```
IQR = Q3 - Q1
```

It captures the **middle half of the data**.

---

## 6️⃣ How IQR Detects Outliers (Logic)

Instead of using mean and standard deviation, IQR defines **acceptable data range**.

### Lower Bound

```
Lower = Q1 - 1.5 × IQR
```

### Upper Bound

```
Upper = Q3 + 1.5 × IQR
```

📌 Any value **outside this range** is an **outlier**.

---

## 7️⃣ Step-by-Step IQR Calculation (Manual Example)

### Dataset

```
Data = [10, 12, 14, 15, 18, 20, 22, 100]
```

### Step 1: Sort Data

```
[10, 12, 14, 15, 18, 20, 22, 100]
```

### Step 2: Find Quartiles

* Q1 = 13 (25th percentile)
* Q3 = 21 (75th percentile)

### Step 3: Compute IQR

```
IQR = 21 - 13 = 8
```

### Step 4: Compute Bounds

```
Lower = 13 - 1.5×8 = 1
Upper = 21 + 1.5×8 = 33
```

### Step 5: Identify Outliers

* `100 > 33` → ❌ Outlier

---

## 8️⃣ Visual Interpretation (Boxplot Logic)

A **boxplot** visually represents:

* Box → Q1 to Q3 (IQR)
* Line inside → Median
* Dots outside → Outliers

📌 IQR is the **foundation** of boxplots.

---

## 9️⃣ IQR Outlier Detection Formula

```
If x < Q1 - 1.5×IQR OR x > Q3 + 1.5×IQR
→ x is an outlier
```

---

## 🔟 Python Implementation (From Scratch)

```python
import numpy as np

data = np.array([10, 12, 14, 15, 18, 20, 22, 100])

Q1 = np.percentile(data, 25)
Q3 = np.percentile(data, 75)
IQR = Q3 - Q1

lower_bound = Q1 - 1.5 * IQR
upper_bound = Q3 + 1.5 * IQR

outliers = data[(data < lower_bound) | (data > upper_bound)]

print("Outliers:", outliers)
```

---

## 1️⃣1️⃣ IQR Using Pandas

```python
import pandas as pd

df = pd.DataFrame({'Salary': [25000, 30000, 28000, 27000, 26000, 250000]})

Q1 = df['Salary'].quantile(0.25)
Q3 = df['Salary'].quantile(0.75)
IQR = Q3 - Q1

lower = Q1 - 1.5 * IQR
upper = Q3 + 1.5 * IQR

outliers = df[(df['Salary'] < lower) | (df['Salary'] > upper)]
print(outliers)
```

---

## 1️⃣2️⃣ Real Dataset Example

### Use Case: Customer Spending

```python
import seaborn as sns

df = sns.load_dataset('tips')

Q1 = df['total_bill'].quantile(0.25)
Q3 = df['total_bill'].quantile(0.75)
IQR = Q3 - Q1

outliers = df[(df['total_bill'] < Q1 - 1.5*IQR) |
              (df['total_bill'] > Q3 + 1.5*IQR)]

print(len(outliers))
```

---

## 1️⃣3️⃣ How to Handle Outliers After Detection

### Option 1: Remove

```python
df_clean = df[(df['total_bill'] >= lower) & (df['total_bill'] <= upper)]
```

### Option 2: Cap (Winsorization)

```python
df['total_bill'] = df['total_bill'].clip(lower, upper)
```

### Option 3: Log Transformation

```python
import numpy as np
df['log_bill'] = np.log(df['total_bill'])
```

---

## 1️⃣4️⃣ IQR vs Z-Score

| Feature                     | IQR | Z-Score |
| --------------------------- | --- | ------- |
| Assumes Normal Distribution | ❌   | ✅       |
| Robust to Outliers          | ✅   | ❌       |
| Works with Skewed Data      | ✅   | ❌       |
| Uses Mean & Std             | ❌   | ✅       |

📌 **IQR is preferred in real-world datasets**.

---

## 1️⃣5️⃣ When to Use IQR

✅ Use IQR when:

* Data is skewed
* Distribution is unknown
* Dataset has extreme values
* Robust preprocessing is needed

---

## 1️⃣6️⃣ Common Mistakes & Edge Cases

❌ Applying IQR blindly on categorical data
❌ Removing valid extreme values (domain knowledge ignored)
❌ Using IQR on very small datasets

---

## 1️⃣7️⃣ IQR in Machine Learning Pipelines

```python
from sklearn.preprocessing import FunctionTransformer

IQR_transformer = FunctionTransformer(
    lambda x: x.clip(x.quantile(0.25) - 1.5*(x.quantile(0.75)-x.quantile(0.25)),
                     x.quantile(0.75) + 1.5*(x.quantile(0.75)-x.quantile(0.25)))
)
```

---

## 1️⃣8️⃣ Interview Questions

* Why is IQR better than Z-score?
* What does 1.5 × IQR mean?
* Can IQR remove valid extreme values?
* Is IQR suitable for time series data?

---

## 1️⃣9️⃣ Summary Cheat Sheet

| Concept     | Key Point                |
| ----------- | ------------------------ |
| IQR         | Q3 − Q1                  |
| Lower Bound | Q1 − 1.5×IQR             |
| Upper Bound | Q3 + 1.5×IQR             |
| Best For    | Skewed & real-world data |

---

## ✅ Final Takeaway

**IQR is one of the safest, simplest, and most powerful techniques for outlier detection in data science.**

It should be your **default choice** unless strong assumptions justify otherwise.

---

📥 **This README is fully GitHub-ready.** You can directly download and use it in your projects.

If you want:

* Visual diagrams added
* sklearn-compatible transformer
* Real ML case study

Just tell me 👍
