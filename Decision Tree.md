# Decision Tree – Complete Guide

A **Decision Tree** is a supervised machine learning algorithm used for **classification** and **regression**. It models decisions using a tree-like structure where data is split step-by-step based on feature values to reach a final prediction.

---

## 1. Why Decision Trees?

Decision Trees are popular because:

* Easy to **understand and interpret**
* Require **little data preprocessing**
* Can handle **numerical and categorical data**
* Work for **non‑linear relationships**
* Mimic human decision-making

However, they can **overfit** if not properly controlled.

---

## 2. Basic Terminology

| Term                 | Meaning                                   |
| -------------------- | ----------------------------------------- |
| Root Node            | Topmost node where the first split occurs |
| Internal Node        | A node that splits into further nodes     |
| Leaf / Terminal Node | Final output (class or value)             |
| Splitting            | Dividing data based on a feature          |
| Parent / Child       | Nodes before and after a split            |
| Depth                | Longest path from root to leaf            |
| Impurity             | Measure of how mixed the data is          |

---

## 3. How a Decision Tree Works (Intuition)

1. Start with all data at the root
2. Try all features and possible split points
3. Measure **impurity reduction** after each split
4. Choose the split that gives the **best purity improvement**
5. Repeat recursively until a stopping condition is met

---

## 4. Types of Decision Trees

### 4.1 Classification Tree

* Target variable is **categorical**
* Uses **Entropy, Gini Index, Chi‑Square**

Example: Spam / Not Spam

### 4.2 Regression Tree

* Target variable is **continuous**
* Uses **Mean Squared Error (MSE)** or **Variance Reduction**

Example: House Price Prediction

---

## 5. Impurity Measures (Splitting Criteria)

### 5.1 Entropy

Entropy measures **randomness or disorder**.

#### Formula

Entropy(S) = − Σ pᵢ log₂(pᵢ)

Where pᵢ is the probability of class i.

#### Example

Suppose a dataset:

* 9 Yes
* 5 No

p(Yes) = 9/14
p(No) = 5/14

Entropy = −(9/14 log₂ 9/14 + 5/14 log₂ 5/14)
Entropy ≈ 0.94

Higher entropy → more impurity
Lower entropy → purer node

---

### 5.2 Information Gain

Information Gain tells **how much entropy decreases** after a split.

#### Formula

IG(S, A) = Entropy(S) − Σ (|Sᵥ| / |S|) × Entropy(Sᵥ)

Where:

* S = parent set
* Sᵥ = subset after split on feature A

#### Example

If Entropy(parent) = 0.94
After split, weighted entropy = 0.69

Information Gain = 0.94 − 0.69 = 0.25

Higher IG → better split

---

### 5.3 Gini Index (Detailed Explanation)

The **Gini Index** measures how **impure** or **mixed** a node is. It represents the probability that a randomly chosen data point would be **incorrectly classified** if it were labeled according to the class distribution of that node.

In simple words:

* **Lower Gini** → Node is purer (mostly one class)
* **Higher Gini** → Node is more mixed

---

#### Mathematical Formula

Gini(S) = 1 − Σ (pᵢ)²

Where:

* S = dataset at a node
* pᵢ = proportion of class i in the dataset

---

#### Step-by-Step Numerical Example

Suppose we have a node with 10 samples:

* Class A = 7
* Class B = 3

Probabilities:

* p(A) = 7/10 = 0.7
* p(B) = 3/10 = 0.3

Gini(S) = 1 − (0.7)² − (0.3)²
Gini(S) = 1 − 0.49 − 0.09
Gini(S) = **0.42**

This means there is a **42% chance** of misclassification at this node.

---

#### Gini for a Split (Weighted Gini)

When splitting a node, we calculate **weighted Gini**:

Gini_split = Σ (|Sᵥ| / |S|) × Gini(Sᵥ)

Where:

* S = parent node
* Sᵥ = child nodes after split

The split with **lowest weighted Gini** is selected.

---

#### Example of Split Selection

Parent Gini = 0.50

After split:

* Left child Gini = 0.20 (6 samples)
* Right child Gini = 0.30 (4 samples)

Weighted Gini = (6/10 × 0.20) + (4/10 × 0.30)
Weighted Gini = 0.12 + 0.12 = **0.24**

This split significantly improves purity.

---

#### Why CART Uses Gini

* Computationally **faster** than entropy
* No logarithms involved
* Performs very similarly to entropy in practice

---

### 5.4 Chi-Square (χ²) Split (Detailed Explanation)

The **Chi-Square test** is a **statistical hypothesis test** used to determine whether a **feature and target variable are independent**.

In decision trees, Chi-Square helps answer:

> "Does this feature really influence the target, or is the split happening by chance?"

---

#### Hypotheses

* **Null Hypothesis (H₀):** Feature and target are independent
* **Alternative Hypothesis (H₁):** Feature and target are dependent

If the null hypothesis is rejected, the feature is useful for splitting.

---

#### Chi-Square Formula

χ² = Σ ( (Observed − Expected)² / Expected )

Where:

* Observed = actual class counts after split
* Expected = class counts assuming no relationship

---

#### Step-by-Step Example

Suppose we want to check if **Gender** affects **Loan Approval**.

Observed Data:

| Gender | Approved | Rejected | Total |
| ------ | -------- | -------- | ----- |
| Male   | 30       | 10       | 40    |
| Female | 10       | 20       | 30    |
| Total  | 40       | 30       | 70    |

---

#### Step 1: Calculate Expected Values

Expected = (Row Total × Column Total) / Grand Total

Example:
Expected(Male, Approved) = (40 × 40) / 70 ≈ 22.86

Similarly calculate for all cells.

---

#### Step 2: Compute χ² Value

χ² = Σ ( (O − E)² / E )

After calculation, assume:
χ² = **15.3**

---

#### Step 3: Decision Rule

* Higher χ² value → stronger dependency
* Compare with critical value (or use p-value)

If χ² is high, the feature is **important**.

---

#### How Chi-Square Is Used in Trees

* Features with **high χ²** are preferred
* Features with **low χ²** may be discarded
* Often combined with a **significance level** (α = 0.05)

---

#### When Chi-Square Is Useful

* Categorical features
* Large datasets
* Feature selection before tree building

---

#### Limitations

* Not suitable for continuous features (without binning)
* Sensitive to sample size
* Less common in modern tree libraries

---

-|----|----|
| Computation | Log based | Squared based |
| Speed | Slower | Faster |
| Used in | ID3, C4.5 | CART |

---

### 5.4 Chi‑Square (χ²) Split

Chi‑Square checks **statistical dependency** between feature and target.

#### Formula

χ² = Σ ( (Observed − Expected)² / Expected )

* Higher χ² → feature is more relevant
* Used mainly for **categorical features**

#### Example Concept

If observed class distribution after split differs significantly from expected distribution, the feature is useful.

---

## 6. Splitting Numerical Features

For numerical features:

1. Sort unique values
2. Try midpoints as thresholds
3. Evaluate impurity for each threshold
4. Choose best threshold

Example:

Age ≤ 30 vs Age > 30

---

## 7. Stopping Criteria

Tree growth stops when:

* All samples belong to one class
* Maximum depth reached
* Minimum samples per split reached
* No impurity reduction

---

## 8. Overfitting in Decision Trees

A deep tree:

* Memorizes training data
* Performs poorly on new data

Solution → **Pruning**

---

## 9. Pruning in Decision Trees

Pruning removes unnecessary branches to improve **generalization**.

---

### 9.1 Pre‑Pruning (Early Stopping)

Stop tree growth early using constraints.

#### Techniques

| Parameter         | Meaning                |
| ----------------- | ---------------------- |
| max_depth         | Maximum depth of tree  |
| min_samples_split | Min samples to split   |
| min_samples_leaf  | Min samples in leaf    |
| max_features      | Max features per split |

#### Example Code

```python
from sklearn.tree import DecisionTreeClassifier

model = DecisionTreeClassifier(
    criterion="gini",
    max_depth=4,
    min_samples_split=10,
    min_samples_leaf=5
)
model.fit(X_train, y_train)
```

Pros:

* Faster
* Simple

Cons:

* Might underfit

---

### 9.2 Post‑Pruning (Cost Complexity Pruning)

Tree grows fully first, then pruned.

#### Cost Complexity Formula

Rα(T) = R(T) + α × |T|

Where:

* R(T) = error
* |T| = number of leaves
* α = complexity parameter

#### Code Example

```python
from sklearn.tree import DecisionTreeClassifier
from sklearn.model_selection import cross_val_score

path = model.cost_complexity_pruning_path(X_train, y_train)
ccp_alphas = path.ccp_alphas

scores = []
for alpha in ccp_alphas:
    clf = DecisionTreeClassifier(ccp_alpha=alpha)
    score = cross_val_score(clf, X_train, y_train, cv=5).mean()
    scores.append(score)

best_alpha = ccp_alphas[scores.index(max(scores))]

final_model = DecisionTreeClassifier(ccp_alpha=best_alpha)
final_model.fit(X_train, y_train)
```

Pros:

* Better generalization

Cons:

* Computationally expensive

---

## 10. Regression Trees

### Splitting Criterion

Variance Reduction:

Var(S) − Σ (|Sᵥ| / |S|) × Var(Sᵥ)

#### Example Code

```python
from sklearn.tree import DecisionTreeRegressor

reg = DecisionTreeRegressor(max_depth=5)
reg.fit(X_train, y_train)
```

---

## 11. Bias‑Variance Tradeoff

| Tree Type    | Bias | Variance |
| ------------ | ---- | -------- |
| Shallow Tree | High | Low      |
| Deep Tree    | Low  | High     |

Pruning balances this tradeoff.

---

## 12. Advantages & Disadvantages

### Advantages

* Interpretable
* Handles missing values
* No scaling required

### Disadvantages

* Overfitting
* Unstable to small data changes
* Greedy algorithm

---

## 13. Decision Tree Algorithms

| Algorithm | Split Criterion  |
| --------- | ---------------- |
| ID3       | Information Gain |
| C4.5      | Gain Ratio       |
| CART      | Gini / MSE       |

---

## 14. Feature Importance

Decision Trees calculate feature importance based on **impurity reduction**.

```python
model.feature_importances_
```

---

## 15. When to Use Decision Trees

Use when:

* Interpretability is required
* Data is non‑linear
* Feature interactions are important

Avoid when:

* Dataset is very small
* High stability is required

---

## 16. Final Notes

Decision Trees are the **foundation** of powerful models like:

* Random Forest
* Gradient Boosting
* XGBoost

Mastering Decision Trees is essential for advanced ML.

---

**End of Document**
