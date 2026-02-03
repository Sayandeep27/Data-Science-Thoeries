# Hyperparameter Optimization in Machine Learning

## GridSearchCV, Parameterized Cross-Validation, and Optuna (Complete Guide)

---

## 📌 Introduction

In Machine Learning, **hyperparameters** are configuration values that are **set before training** a model. Unlike model parameters (weights), hyperparameters control *how* the model learns.

Examples:

* Number of trees in Random Forest
* Learning rate in Gradient Boosting
* C and gamma in SVM

Choosing the right hyperparameters can:

* Improve accuracy
* Reduce overfitting
* Improve generalization

This README explains **three major approaches**:

1. **GridSearchCV** (Exhaustive Search)
2. **Parameterized Cross-Validation** (Manual + Pipelines)
3. **Optuna** (Smart, Bayesian Optimization)

All concepts are explained in **easy language**, with **theory + diagrams (conceptual) + full Python code**.

---

## 🔍 What is Hyperparameter Tuning?

Hyperparameter tuning is the process of **finding the best combination of hyperparameters** that gives the best model performance on unseen data.

### Why not use default values?

| Problem              | Effect                     |
| -------------------- | -------------------------- |
| Poor hyperparameters | Underfitting / Overfitting |
| Wrong learning rate  | Model fails to converge    |
| Too many trees       | Slow + overfitting         |

---

## 🔁 Cross-Validation (Quick Recap)

Before tuning, understand **cross-validation**.

### K-Fold Cross Validation

1. Split data into K folds
2. Train on K-1 folds
3. Validate on remaining fold
4. Repeat K times
5. Average the scores

This gives **reliable performance estimates**.

---

# 1️⃣ GridSearchCV (Exhaustive Hyperparameter Search)

---

## 🔹 What is GridSearchCV?

**GridSearchCV** tries **all possible combinations** of hyperparameters using cross-validation.

> "Brute-force but reliable"

---

## 🔹 How GridSearchCV Works

1. Define parameter grid
2. Create all combinations
3. Perform CV for each combination
4. Select best parameters

---

## 🔹 Example Parameters Grid

```python
param_grid = {
    'n_estimators': [50, 100, 200],
    'max_depth': [None, 10, 20],
    'min_samples_split': [2, 5]
}
```

Total combinations = 3 × 3 × 2 = **18 models**

---

## 🔹 Full Code Example: GridSearchCV

```python
from sklearn.datasets import load_iris
from sklearn.model_selection import GridSearchCV
from sklearn.ensemble import RandomForestClassifier

# Load data
X, y = load_iris(return_X_y=True)

# Model
model = RandomForestClassifier(random_state=42)

# Parameter Grid
param_grid = {
    'n_estimators': [50, 100],
    'max_depth': [None, 10],
    'min_samples_split': [2, 5]
}

# Grid Search
grid = GridSearchCV(
    estimator=model,
    param_grid=param_grid,
    cv=5,
    scoring='accuracy',
    n_jobs=-1
)

# Fit
grid.fit(X, y)

# Results
print("Best Parameters:", grid.best_params_)
print("Best CV Score:", grid.best_score_)
```

---

## 🔹 Important Attributes

| Attribute      | Meaning              |
| -------------- | -------------------- |
| `best_params_` | Best hyperparameters |
| `best_score_`  | Best CV score        |
| `cv_results_`  | All trial results    |

---

## 🔹 Advantages

✔ Simple
✔ Exhaustive
✔ Deterministic

## 🔹 Disadvantages

❌ Very slow for large grids
❌ Computationally expensive

---

# 2️⃣ Parameterized Cross-Validation (Manual + Pipeline Based CV)

---

## 🔹 What is Parameterized CV?

It means:

* You **manually control hyperparameters**
* Combine **Pipeline + Cross-Validation**
* Useful when logic is complex

Often used in **production ML systems**.

---

## 🔹 Why Use Pipelines?

Pipelines ensure:

* No data leakage
* Proper CV
* Clean workflow

---

## 🔹 Pipeline with Cross Validation

```python
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.model_selection import cross_val_score

pipe = Pipeline([
    ('scaler', StandardScaler()),
    ('model', SVC())
])

scores = cross_val_score(
    pipe,
    X,
    y,
    cv=5,
    scoring='accuracy'
)

print("Mean CV Accuracy:", scores.mean())
```

---

## 🔹 Manual Parameter Looping

```python
params = [
    {'C': 0.1, 'kernel': 'linear'},
    {'C': 1, 'kernel': 'rbf'},
    {'C': 10, 'kernel': 'rbf'}
]

best_score = 0
best_param = None

for p in params:
    pipe.set_params(model__C=p['C'], model__kernel=p['kernel'])
    score = cross_val_score(pipe, X, y, cv=5).mean()
    
    if score > best_score:
        best_score = score
        best_param = p

print(best_param, best_score)
```

---

## 🔹 When to Use Parameterized CV

✔ Custom logic
✔ Conditional parameters
✔ Experiment tracking

---

# 3️⃣ Optuna (Smart Hyperparameter Optimization)

---

## 🔹 What is Optuna?

**Optuna** is a modern **Bayesian optimization framework**.

> "Try fewer experiments, learn from previous ones"

It uses:

* TPE (Tree-structured Parzen Estimator)
* Pruning
* Intelligent sampling

---

## 🔹 Why Optuna is Powerful

| Feature         | Benefit                |
| --------------- | ---------------------- |
| Bayesian Search | Fewer trials           |
| Pruning         | Stops bad trials early |
| Visualization   | Easy analysis          |

---

## 🔹 Optuna Workflow

1. Define objective function
2. Suggest hyperparameters
3. Train model
4. Return score

---

## 🔹 Install Optuna

```bash
pip install optuna
```

---

## 🔹 Full Optuna Example

```python
import optuna
from sklearn.model_selection import cross_val_score
from sklearn.ensemble import RandomForestClassifier


def objective(trial):
    n_estimators = trial.suggest_int('n_estimators', 50, 300)
    max_depth = trial.suggest_int('max_depth', 2, 20)
    min_samples_split = trial.suggest_int('min_samples_split', 2, 10)

    model = RandomForestClassifier(
        n_estimators=n_estimators,
        max_depth=max_depth,
        min_samples_split=min_samples_split,
        random_state=42
    )

    score = cross_val_score(model, X, y, cv=5, scoring='accuracy').mean()
    return score

study = optuna.create_study(direction='maximize')
study.optimize(objective, n_trials=50)

print("Best Params:", study.best_params)
print("Best Score:", study.best_value)
```

---

## 🔹 Optuna Pruning Example

```python
from optuna.integration import SklearnPruningCallback

study = optuna.create_study(
    direction='maximize',
    pruner=optuna.pruners.MedianPruner()
)
```

---

## 🔹 Visualization (Optional)

```python
optuna.visualization.plot_optimization_history(study)
optuna.visualization.plot_param_importances(study)
```

---

# 🔄 Comparison Summary

| Feature     | GridSearchCV | Parameterized CV | Optuna       |
| ----------- | ------------ | ---------------- | ------------ |
| Search Type | Exhaustive   | Manual           | Bayesian     |
| Speed       | Slow         | Medium           | Fast         |
| Scalability | Poor         | Medium           | Excellent    |
| Automation  | High         | Low              | Very High    |
| Best for    | Small grids  | Custom logic     | Large models |

---

## 🧠 When Should You Use What?

* **Learning / Small Dataset** → GridSearchCV
* **Production Pipelines** → Parameterized CV
* **Large ML / Deep Learning** → Optuna

---

## 📦 Best Practices

✔ Always use cross-validation
✔ Combine with pipelines
✔ Track experiments
✔ Avoid data leakage
✔ Prefer Optuna for large spaces

---

## ✅ Final Takeaway

Hyperparameter tuning is **mandatory** for high-quality ML models.

* GridSearchCV = simple but slow
* Parameterized CV = flexible
* Optuna = smart and scalable

---

## ⭐ If this helped you

Give this repo a ⭐ and use it as your **ML interview + production reference**.

Happy Learning 🚀
