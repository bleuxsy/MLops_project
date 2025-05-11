import pandas as pd
from supervised.automl import AutoML

data = pd.read_csv('diabetes_prediction_dataset.csv')
X = data[['bmi', 'HbA1c_level', 'blood_glucose_level']]
y = data['diabetes']

validation = {
    "validation_type": "kfold",
    "k_folds": 5,
    "shuffle": True,
    "stratify": True,
    "random_seed": 42
}

automl_rf = AutoML(
    mode='Explain',
    algorithms=['Random Forest'],
    train_ensemble=False,
    stack_models=False,
    eval_metric='auc',
    validation_strategy=validation,
    explain_level=2
)
automl_rf.fit(X, y)
print("\n[Random Forest]")
leaderboard_rf = automl_rf.get_leaderboard()
print(leaderboard_rf[["model_type", "metric_value", "train_time"]])

automl_xgb = AutoML(
    mode='Explain',
    algorithms=['Xgboost'],
    train_ensemble=False,
    stack_models=False,
    eval_metric='auc',
    validation_strategy=validation,
    explain_level=2
)
automl_xgb.fit(X, y)
print("\n[Xgboost]")
leaderboard_xgb = automl_xgb.get_leaderboard()
print(leaderboard_xgb[["model_type", "metric_value", "train_time"]])

automl_nn = AutoML(
    mode='Explain',
    algorithms=['Neural Network'],
    train_ensemble=False,
    stack_models=False,
    eval_metric='auc',
    validation_strategy=validation,
    explain_level=2
)
automl_nn.fit(X, y)
print("\n[Neural Network]")
leaderboard_nn = automl_nn.get_leaderboard()
print(leaderboard_nn[["model_type", "metric_value", "train_time"]])
