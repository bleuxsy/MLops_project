# 🩺 Diabetes Prediction Using AutoML (MLjar)

## 📌 Project Description

This project aims to build a **binary classification model** to predict whether a patient has diabetes (`1`) or not (`0`) using three key health-related features:
- Body Mass Index (BMI)  
- HbA1c level  
- Blood glucose level  

Two approaches are used:
- `manual_model.py`: Manually implemented logistic regression model (Assignment #1)
- `automl_model.py`: Automated model selection using **MLjar-supervised AutoML** (Assignment #2)

The primary objective is to find a model with **high recall**, which is especially important in medical contexts to minimize missed diagnoses of diabetic patients.

---

## 🧪 Models Evaluated (AutoML)

Using MLjar’s `Explain` mode with 5-fold cross-validation, the following models were evaluated:
- Random Forest  
- XGBoost  
- Neural Network  

### 🎯 Key Evaluation Metrics
- Accuracy  
- Precision  
- **Recall** *(critical for this task)*  
- F1-score  
- AUC  
- Confusion Matrix  

> 📌 **Recall** was prioritized due to its significance in correctly identifying diabetic patients and avoiding false negatives.

---

## ⚙️ Development Environment

- **OS**: Ubuntu 24.04 (AWS EC2)  
- **Instance**: t2.micro (1 vCPU)  
- **Storage**: EBS volume (20 GiB)  
- **Python**: 3.12.3 with `venv` virtual environment  
- **Access**: Remote via SSH from macOS  

All required packages are listed in [`requirements.txt`](./requirements.txt)

---

## 🚀 How to Run

```bash
# 1. Install dependencies
pip install -r requirements.txt

# 2. Run the manual model (logistic regression)
python manual_model.py

# 3. Run the AutoML training script
python automl_model.py