# Machine Learning Error Analysis & Evaluation Metrics

This document provides an overview of key concepts in error analysis and model evaluation.

---

## 📊 Confusion Matrix
A **confusion matrix** is a table that summarizes how well a classification model performs by comparing the predicted labels with the actual labels.

|                | Predicted Positive | Predicted Negative |
|----------------|--------------------|--------------------|
| **Actual Positive** | True Positive (TP)  | False Negative (FN) |
| **Actual Negative** | False Positive (FP) | True Negative (TN)  |

- **Rows = Actual classes**  
- **Columns = Predicted classes**  
- **Diagonal = Correct predictions**

---

## ❌ Type I & Type II Errors
- **Type I error (False Positive):** Predicting positive when it is actually negative.  
- **Type II error (False Negative):** Predicting negative when it is actually positive.  

👉 Think: **Type I = False Alarm**, **Type II = Missed Detection**

---

## 📈 Sensitivity, Specificity, Precision & Recall

- **Sensitivity (Recall, True Positive Rate):**  
  \[
  \text{Sensitivity} = \frac{TP}{TP + FN}
  \]  
  → “Of all actual positives, how many did we catch?”

- **Specificity (True Negative Rate):**  
  \[
  \text{Specificity} = \frac{TN}{TN + FP}
  \]  
  → “Of all actual negatives, how many did we correctly reject?”

- **Precision (Positive Predictive Value):**  
  \[
  \text{Precision} = \frac{TP}{TP + FP}
  \]  
  → “When we predict positive, how often are we right?”

- **Recall:** Another name for **Sensitivity**.

---

## ⚖️ F1 Score
The **F1 score** balances Precision and Recall using their harmonic mean:

\[
F1 = \frac{2 \cdot (Precision \cdot Recall)}{Precision + Recall}
\]

- High only when **both** precision and recall are high.
- Useful for imbalanced datasets.

---

## 🧠 Bias and Variance

- **Bias:**  
  Systematic error from a model being too simple or making wrong assumptions.  
  → High bias = underfitting.

- **Variance:**  
  Error from a model being too sensitive to training data.  
  → High variance = overfitting.

- **Bias–Variance Tradeoff:**  
  - High Bias, Low Variance → underfit.  
  - Low Bias, High Variance → overfit.  
  - Low Bias, Low Variance → ideal.  
  - High Bias, High Variance → worst of both worlds.

---

## 🔎 Irreducible Error
The portion of error that cannot be eliminated, even with a perfect model, due to noise or randomness in the data.

---

## 📉 Bayes Error
- The **lowest possible error** for any classifier on a given problem.  
- Even the best model cannot do better than Bayes error.

---

## 🧮 Approximating Bayes Error
- Estimate it using **human-level performance** or **expert benchmarks**.  
- If human error is ~14%, Bayes error is assumed close to that.

---

## 🧾 Calculating Bias and Variance
- **Bias:** Compare training error to Bayes error (or human-level error).  
  - High training error → High bias.
- **Variance:** Compare validation error to training error.  
  - Large gap → High variance.

---

## 🏗️ Creating a Confusion Matrix

### From one-hot labels (NumPy example)
```python
import numpy as np

def create_confusion_matrix(labels, logits):
    true_classes = np.argmax(labels, axis=1)
    pred_classes = np.argmax(logits, axis=1)

    classes = labels.shape[1]
    confusion = np.zeros((classes, classes), dtype=int)

    for t, p in zip(true_classes, pred_classes):
        confusion[t, p] += 1

    return confusion
```
### Example

```python
labels = np.array([[1,0,0],
                   [0,1,0],
                   [0,0,1],
                   [1,0,0]])

logits = np.array([[1,0,0],
                   [0,0,1],
                   [0,0,1],
                   [0,1,0]])

print(create_confusion_matrix(labels, logits))
```
**Output:**
```python
[[1 1 0]
 [0 0 1]
 [0 0 1]]
```

---

✅ Summary

- **Confusion Matrix** = full view of classification performance.

- **Type I vs II** = false alarm vs missed detection.

- **Sensitivity, Specificity, Precision, Recall** = different ways to measure correctness.

- **F1 Score** = balances precision & recall.

- **Bias & Variance** = model vs data error sources.

- **Irreducible/Bayes Error** = limits of performance.

- **Bias vs Variance calculations** = compare training, validation, and human-level performance.