# Diabetes Prediction Using AutoML (MLjar)

## 📌 Project Description

This project aims to build a **binary classification model** that predicts whether a patient has diabetes (1) or not (0), based on three key health features:
- Body Mass Index (BMI)
- HbA1c level
- Blood glucose level

To automate model selection and evaluation, we used **MLjar-supervised**, a powerful AutoML framework. The goal was to identify the most suitable model with **high recall**, which is critical in medical diagnostics to avoid missed diagnoses.

---

## 🧪 Models Evaluated
The following models were trained and evaluated using the `Explain` mode of MLjar with 5-fold cross-validation:
- Random Forest
- XGBoost
- Neural Network

### 🎯 Key Evaluation Metrics
- Accuracy
- Precision
- Recall
- F1-score
- AUC
- Confusion Matrix

> 📌 **Recall** was treated as the most important metric due to the medical nature of the task.

---

## ⚙️ Development Environment

- **OS**: Ubuntu 24.04 (AWS EC2)
- **Instance**: t2.micro (1 vCPU)
- **Storage**: EBS volume (20 GiB)
- **Python**: 3.12.3 with `venv` virtual environment
- **Access**: SSH from macOS Terminal

All required libraries and versions are listed in [`requirements.txt`](./requirements.txt)

---

## 🚀 How to Run

```bash
# Install required packages
pip install -r requirements.txt

# Run the AutoML training script
python automl_multiple.py