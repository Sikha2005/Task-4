# Task-4
# 📊 Logistic Regression Classifier — Breast Cancer Detection

## 📌 Objective:
To build a *binary classifier* using *Logistic Regression* for classifying breast cancer tumors as *Malignant (M)* or *Benign (B)* using the Breast Cancer Wisconsin dataset.

---

## 📦 Tools & Libraries:
- *Python 3.x*
- *Pandas*
- *NumPy*
- *Matplotlib*
- *Scikit-learn*

---

## 📁 Dataset:
- *File:* data.csv
- *Source:* Breast Cancer Wisconsin dataset  
- *Target Column:* diagnosis
  - M → Malignant (1)
  - B → Benign (0)

---

## 📊 Task Steps:
1. Load and preprocess the dataset:
   - Drop unnecessary columns (id, Unnamed: 32).
   - Encode target variable.
2. Split data into *training* and *testing* sets.
3. Standardize features using StandardScaler.
4. Fit a *Logistic Regression* model.
5. Evaluate model performance:
   - *Confusion Matrix*
   - *Precision, Recall, F1-Score*
   - *ROC-AUC Score*
   - *ROC Curve plot*
6. Demonstrate how the *Sigmoid Function* maps outputs between 0 and 1.
7. Adjust classification threshold and observe changes in the confusion matrix.

---

## 📈 Performance Summary:
- *ROC-AUC Score:* ~0.996  
- *Accuracy:* ~96%
- Very good balance between *precision* and *recall*.

---

THE OUTPUT BY RUNNING THE PROGRAM US GIVEN BELOW:

Confusion Matrix:

 [71  1]
 
 [ 3 39]

Classification Report:
               precision    recall  f1-score   support

           0       0.96      0.99      0.97        72
           1       0.97      0.93      0.95        42

    accuracy                           0.96       114
   macro avg       0.97      0.96      0.96       114
weighted avg       0.97      0.96      0.96       114

ROC-AUC Score: 0.996
