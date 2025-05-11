import pandas as pd
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split, KFold
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import precision_score, recall_score, f1_score, roc_auc_score
import matplotlib.pyplot as plt


df = pd.read_csv('diabetes_prediction_dataset.csv')


features = ['bmi', 'HbA1c_level', 'blood_glucose_level']
X = df[features]
y = df['diabetes']


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.25, random_state=0)


scaler = StandardScaler()
X_train = scaler.fit_transform(X_train)
X_test = scaler.transform(X_test)

model = LogisticRegression().fit(X_train, y_train)


accuracy = model.score(X_test, y_test)
print("\n [모델 성능]")
print(f'Testing Accuracy: {accuracy*100:.2f}%')

coefficients = model.coef_[0]
feature_df = pd.DataFrame({'Feature': features, 'Coefficient': coefficients})
print()
print(feature_df)
print()
print("\n [5-Fold Cross Validation 결과]")

kf = KFold(n_splits=5, shuffle=True, random_state=42)
precision_list, recall_list, f1_list, auc_list = [], [], [], []

for fold_idx, (train_idx, val_idx) in enumerate(kf.split(X)):
    X_train_k, X_val_k = X.iloc[train_idx], X.iloc[val_idx]
    y_train_k, y_val_k = y.iloc[train_idx], y.iloc[val_idx]

    
    scaler_k = StandardScaler()
    X_train_k_scaled = scaler_k.fit_transform(X_train_k)
    X_val_k_scaled = scaler_k.transform(X_val_k)

    
    model_k = LogisticRegression(max_iter=1000)
    model_k.fit(X_train_k_scaled, y_train_k)

    
    y_pred_k = model_k.predict(X_val_k_scaled)
    y_prob_k = model_k.predict_proba(X_val_k_scaled)[:, 1]

    
    precision = precision_score(y_val_k, y_pred_k)
    recall = recall_score(y_val_k, y_pred_k)
    f1 = f1_score(y_val_k, y_pred_k)
    auc = roc_auc_score(y_val_k, y_prob_k)

    precision_list.append(precision)
    recall_list.append(recall)
    f1_list.append(f1)
    auc_list.append(auc)

    print(f"Fold {fold_idx+1}: Precision={precision:.4f}, Recall={recall:.4f}, F1={f1:.4f}, AUC={auc:.4f}")


print("\n [5-Fold 평균 성능]")
print("Average Precision:", round(np.mean(precision_list), 4))
print("Average Recall:", round(np.mean(recall_list), 4))
print("Average F1 Score:", round(np.mean(f1_list), 4))
print("Average AUC:", round(np.mean(auc_list), 4))
print()



def predict_diabetes(bmi, hba1c, glucose):
    """
    환자의 bmi, HbA1c_level, blood_glucose_level 값을 받아
    당뇨병 여부를 예측하고 결과를 출력.
    """
    new_data = pd.DataFrame([[bmi, hba1c, glucose]], columns=features)
    new_data_scaled = scaler.transform(new_data)
    prediction = model.predict(new_data_scaled)

    
    print(f'입력값: BMI={bmi}, HbA1c={hba1c}, blood_glucose={glucose}')
    print('예측 결과 (Diabetes?):', 'Yes' if prediction[0] == 1 else 'No')

print('\n [환자 시나리오]')
predict_diabetes(26.5, 6.2, 155)
predict_diabetes(32.1, 8.5, 190)
predict_diabetes(22.3, 4.8, 110)


