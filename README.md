

# 🌳 DecisionTree-with-numpy

A custom implementation of a **Multi-class Decision Tree** structure powered by **Binary Logistic Regression** engines. Instead of using traditional Gini impurity or Information Gain, this project builds a hierarchical decision structure using Gradient Descent-based binary classifiers to categorize the Iris dataset.

---

## 🎯 Study Goals
- **Hierarchical Classification**: Understand how to decompose a multi-class problem (3 classes) into a sequence of binary decisions.
- **Logistic Regression from Scratch**: Implement Gradient Descent, Sigmoid activation, and Binary Cross-Entropy Loss using only NumPy.
- **Manual Data Pipeline**: Handle data normalization, shuffling, and splitting without high-level framework utilities.

---

## 🔬 Implementation Details

### 1. The Binary Engine (`Binary_inquiry_train`)
Each node in the tree is a logistic regression model that performs:
- **Normalization**: Standardizes features to improve Gradient Descent convergence.
- **Optimization**: Uses **Gradient Descent (GD)** to minimize Binary Cross-Entropy loss over 8,000 iterations.
- **Prediction**: Outputs a probability via the Sigmoid function, which is then rounded to a binary decision (0 or 1).

### 2. Decision Logic (The "Tree" Structure)
The model classifies Iris species through a two-step inquiry:
1. **First Node**: Decides whether the sample is `Virginica (2)` or `Others (0 or 1)`.
2. **Second Node**: If not Virginica, it further decides between `Setosa (0)` and `Versicolor (1)`.



### 3. Training & Evaluation
- **Dataset**: Scikit-learn's Iris dataset (4 features, 3 target classes).
- **Metric**: Accuracy is calculated by passing test data through the hierarchical inquiry path and comparing it with ground truth labels.

---

## 🛠 Tech Stack
- **Language**: Python 3.x
- **Numerical Processing**: NumPy
- **Progress Tracking**: Tqdm
- **Dataset Source**: Scikit-learn (only for `load_iris` and `shuffle`)

---

## 📂 Project Structure
- `decision_tree_logic.py`: Contains the `Binary_inquiry_train` class and the main execution logic for hierarchical classification.

---

## 📊 How It Works
1. **Train Node 1**: Train a model to separate `{0, 1}` from `{2}`.
2. **Train Node 2**: Train a model to separate `{0}` from `{1}` using a subset of the data.
3. **Inference**:
   - Start at Node 1.
   - If `Predict == 1` → Class 2.
   - If `Predict == 0` → Move to Node 2.
   - Node 2 predicts Class 0 or Class 1.

---


## Data

* Iris

## Comment 
* I was planned to implement DecisionTree. But this is just binary classification iteration for multi classification
* I think the important part of " Decision Tree " is the algorithm how the node choose the feature represented output well. So Decision Tree have to use algorithms such as  "CART algo" or "ID3 algo" or " c4.5 algo"
* The thing I have to do next is remake the Decision Tree using the algorithms I told and "Gini Index" ( or entropy )

![myplot2](myplot2.png)
![myplot](myplot.png)
![11](11.PNG)
