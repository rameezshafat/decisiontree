

---

# Decision Tree Classifier — Pure Python Implementation

A clean, educational implementation of a **Decision Tree Classifier** built entirely from scratch using Python.
This project demonstrates the full workflow behind decision-tree learning, including impurity calculation, optimal split selection, recursive tree construction, and prediction.

---

## 📘 Overview

This implementation includes:

* Gini impurity metric
* Exhaustive search for the best split across all features
* Information gain evaluation
* Recursive tree growth with robust stopping criteria
* Human-readable tree visualization
* Support for prediction on new data

It is designed for learning, experimentation, and teaching machine-learning fundamentals without relying on external ML libraries.

---

## ⚙️ Core Components

### **1. Gini Impurity**

Quantifies label mix:

```
Gini = 1 − Σ(pᵢ²)
```

Used to measure node quality during splitting.

### **2. Best Split Search**

For each feature and unique threshold:

* partition data
* compute weighted impurity
* evaluate information gain
* retain the split with maximum improvement

### **3. Tree Construction**

Nodes are expanded recursively until any stopping condition is met:

* pure labels
* maximum depth reached
* insufficient samples
* zero information gain

Leaves store the majority class.

### **4. Prediction**

Traversal begins at the root and proceeds left/right based on feature thresholds until a leaf is reached.

### **5. Tree Visualization**

A helper provides a readable textual representation of the entire structure.

---

## 🧩 Functions Included

* `gini(y)`
* `best_split(X, y)`
* `build_tree(X, y, max_depth, min_samples_leaf)`
* `predict_single(tree, sample)`
* `predict(tree, X)`
* `print_tree(tree, feature_names)`

All functions are concise and reflect the mechanics of classical decision-tree algorithms.

---

##  Example Usage

```python
tree = build_tree(X, y, max_depth=3, min_samples_leaf=1)
print_tree(tree, ["Feature_0", "Feature_1"])
predictions = predict(tree, X)
```

---

## 🔧 Hyperparameters

| Parameter          | Description                                         |
| ------------------ | --------------------------------------------------- |
| `max_depth`        | Limits tree depth to prevent overfitting            |
| `min_samples_leaf` | Ensures leaves contain enough samples for stability |

---


