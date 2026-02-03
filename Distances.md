# 📏 Distance Metrics in Machine Learning – Complete Beginner-to-Advanced Guide

A **professional, GitHub-ready README.md** explaining **all major distance types**, with **deep focus on Euclidean and Manhattan distance**, written in **easy language**, with **examples, intuition, math, visuals (explained), and Python code**.

---

## 📌 Table of Contents

1. What is Distance in Machine Learning?
2. Why Distance Metrics Matter
3. Types of Distance Metrics (Overview)
4. Euclidean Distance (In-Depth)
5. Manhattan Distance (In-Depth)
6. Euclidean vs Manhattan (Comparison)
7. Other Important Distance Metrics

   * Minkowski Distance
   * Chebyshev Distance
   * Cosine Distance
   * Hamming Distance
   * Jaccard Distance
8. Distance Metrics in Real ML Algorithms
9. Feature Scaling and Distance Metrics
10. Choosing the Right Distance Metric
11. Complete Python Code Examples
12. Summary Cheat Sheet

---

## 1️⃣ What is Distance in Machine Learning?

In **machine learning**, *distance* measures **how similar or dissimilar two data points are**.

Think of distance as:

* How close two people live 🏠
* How similar two customers are 🧍🧍
* How alike two images or texts are 🖼️📄

Mathematically, distance converts **difference between feature values** into a **single number**.

---

## 2️⃣ Why Distance Metrics Matter

Distance is the backbone of many ML algorithms:

| Algorithm              | Why Distance is Important     |
| ---------------------- | ----------------------------- |
| KNN                    | Finds nearest neighbors       |
| K-Means                | Forms clusters using distance |
| DBSCAN                 | Density based on distance     |
| Recommendation systems | Similar users/items           |
| Anomaly Detection      | Far = anomaly                 |

⚠️ Wrong distance choice = bad model performance

---

## 3️⃣ Types of Distance Metrics (Overview)

| Distance Type | Best Used For                |
| ------------- | ---------------------------- |
| **Euclidean** | Continuous numeric data      |
| **Manhattan** | Grid-like data, robustness   |
| Minkowski     | Generalized distance         |
| Chebyshev     | Chessboard-like movement     |
| Cosine        | Text & high-dimensional data |
| Hamming       | Categorical / binary data    |
| Jaccard       | Set similarity               |

---

## 4️⃣ Euclidean Distance (⭐ MOST IMPORTANT)

### 🔹 What is Euclidean Distance?

Euclidean distance is the **straight-line distance** between two points.

👉 Same distance you measure using a ruler.

---

### 🔹 Formula

For two points:

A = (x₁, x₂, ..., xₙ)
B = (y₁, y₂, ..., yₙ)

```
Euclidean Distance = √[(x₁−y₁)² + (x₂−y₂)² + ... + (xₙ−yₙ)²]
```

---

### 🔹 Simple 2D Example

Points:

* A = (2, 3)
* B = (6, 7)

Calculation:

```
√[(6−2)² + (7−3)²]
= √(16 + 16)
= √32 ≈ 5.66
```

---

### 🔹 Intuitive Understanding

* Measures **direct shortest path**
* Sensitive to **large differences**
* Works best when features are **scaled**

---

### 🔹 Python Code (Manual)

```python
import math

A = [2, 3]
B = [6, 7]

distance = math.sqrt(sum((a - b) ** 2 for a, b in zip(A, B)))
print(distance)
```

---

### 🔹 Using scikit-learn

```python
from sklearn.metrics.pairwise import euclidean_distances

A = [[2, 3]]
B = [[6, 7]]

print(euclidean_distances(A, B))
```

---

### 🔹 Where Euclidean Distance is Used

* K-Nearest Neighbors (KNN)
* K-Means Clustering
* PCA (distance in feature space)
* Image similarity

---

### ⚠️ Disadvantages

* Very sensitive to **outliers**
* Fails if features are on different scales

---

## 5️⃣ Manhattan Distance (⭐ VERY IMPORTANT)

### 🔹 What is Manhattan Distance?

Manhattan distance measures distance **along axes**, not diagonally.

👉 Like moving in a city with **right-angle streets**.

---

### 🔹 Formula

```
Manhattan Distance = |x₁−y₁| + |x₂−y₂| + ... + |xₙ−yₙ|
```

---

### 🔹 Example

Points:

* A = (2, 3)
* B = (6, 7)

Calculation:

```
|6−2| + |7−3| = 4 + 4 = 8
```

---

### 🔹 Intuition

* Moves **step-by-step**
* Less sensitive to outliers
* Works well for **high-dimensional data**

---

### 🔹 Python Code (Manual)

```python
A = [2, 3]
B = [6, 7]

distance = sum(abs(a - b) for a, b in zip(A, B))
print(distance)
```

---

### 🔹 Using scikit-learn

```python
from sklearn.metrics.pairwise import manhattan_distances

A = [[2, 3]]
B = [[6, 7]]

print(manhattan_distances(A, B))
```

---

### 🔹 Where Manhattan Distance is Used

* KNN (robust version)
* Sparse data
* Recommendation systems
* Text-based feature vectors

---

## 6️⃣ Euclidean vs Manhattan (Comparison)

| Feature                 | Euclidean          | Manhattan     |
| ----------------------- | ------------------ | ------------- |
| Path                    | Straight line      | Grid path     |
| Sensitivity to outliers | High               | Lower         |
| Speed                   | Slower             | Faster        |
| Best for                | Low-dim continuous | High-dim data |
| Uses square             | Yes                | No            |

---

## 7️⃣ Other Important Distance Metrics

### 🔹 Minkowski Distance

Generalized form:

```
D = ( Σ |xᵢ − yᵢ|ᵖ )¹ᐟᵖ
```

* p = 1 → Manhattan
* p = 2 → Euclidean

```python
from scipy.spatial.distance import minkowski
minkowski([2,3], [6,7], p=3)
```

---

### 🔹 Chebyshev Distance

Maximum difference in any dimension.

```python
from scipy.spatial.distance import chebyshev
chebyshev([2,3], [6,7])
```

---

### 🔹 Cosine Distance

Measures **angle**, not magnitude.

```python
from sklearn.metrics.pairwise import cosine_distances
cosine_distances([[1,2,3]], [[2,4,6]])
```

Used heavily in **NLP & embeddings**.

---

### 🔹 Hamming Distance

Counts mismatched positions.

```python
from scipy.spatial.distance import hamming
hamming([1,0,1], [1,1,0])
```

---

### 🔹 Jaccard Distance

Set similarity.

```python
from sklearn.metrics import jaccard_score
jaccard_score([1,0,1], [1,1,0])
```

---

## 8️⃣ Distance Metrics in Real ML Algorithms

| Algorithm       | Distance Used         |
| --------------- | --------------------- |
| KNN             | Euclidean / Manhattan |
| K-Means         | Euclidean             |
| DBSCAN          | Euclidean             |
| Text similarity | Cosine                |
| Binary features | Hamming               |

---

## 9️⃣ Feature Scaling is CRITICAL

Distance metrics are **scale sensitive**.

Example:

* Salary: 50,000
* Age: 25

Salary dominates distance ❌

### ✅ Solution: Scaling

```python
from sklearn.preprocessing import StandardScaler

scaler = StandardScaler()
X_scaled = scaler.fit_transform(X)
```

---

## 🔟 How to Choose the Right Distance Metric

| Data Type            | Recommended Distance |
| -------------------- | -------------------- |
| Numeric continuous   | Euclidean            |
| High-dimensional     | Manhattan            |
| Text / embeddings    | Cosine               |
| Binary / categorical | Hamming              |
| Set-based            | Jaccard              |

---

## 1️⃣1️⃣ Complete Example: KNN with Distance

```python
from sklearn.neighbors import KNeighborsClassifier

knn = KNeighborsClassifier(
    n_neighbors=5,
    metric='euclidean'
)

knn.fit(X_train, y_train)
knn.predict(X_test)
```

---

## 1️⃣2️⃣ Summary Cheat Sheet

| Metric    | Key Idea          |
| --------- | ----------------- |
| Euclidean | Straight-line     |
| Manhattan | Grid-based        |
| Minkowski | Generalized       |
| Cosine    | Angle-based       |
| Hamming   | Position mismatch |
| Jaccard   | Set overlap       |

---

## 🎯 Final Takeaway

> **Distance metrics define how your model understands similarity.**

Choosing the **right distance** + **proper scaling** can drastically improve model performance.

---

📌 **This README is ready to be downloaded, used in GitHub, and extended for interviews, projects, and exams.**
