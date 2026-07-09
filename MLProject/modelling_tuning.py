import pandas as pd
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# Hubungkan ke DagsHub
dagshub.init(repo_owner='aksaex', repo_name='MSML_Proyek_Akhir', mlflow=True)

def train_and_log():
    train = pd.read_csv('heart_preprocessing/train.csv')
    test = pd.read_csv('heart_preprocessing/test.csv')
    
    X_train = train.drop('output', axis=1)
    y_train = train['output']
    X_test = test.drop('output', axis=1)
    y_test = test['output']

    rf = RandomForestClassifier(random_state=42)
    param_grid = {'n_estimators': [50, 100], 'max_depth': [5, 10]}
    grid_search = GridSearchCV(rf, param_grid, cv=3)

    print("Mulai melatih model dan mencari parameter terbaik...")
    
    with mlflow.start_run():
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        mlflow.log_param("best_n_estimators", best_model.n_estimators)
        mlflow.log_param("best_max_depth", best_model.max_depth)
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))

        # KUNCI JAWABAN: Paksa Docker menggunakan Python 3.10.12 yang stabil
        custom_env = {
            "name": "mlflow-env",
            "channels": ["conda-forge"],
            "dependencies": [
                "python=3.10.12",
                "pip",
                {
                    "pip": [
                        "mlflow==2.19.0",
                        "scikit-learn",
                        "pandas",
                        "matplotlib",
                        "seaborn",
                        "dagshub"
                    ]
                }
            ]
        }
        
        # Log model dengan environment yang sudah dikunci
        mlflow.sklearn.log_model(best_model, "random_forest_model", conda_env=custom_env)

        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        importances = best_model.feature_importances_
        plt.figure(figsize=(8,6))
        plt.barh(X_train.columns, importances)
        plt.title("Feature Importance")
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        print("Selesai! Model berhasil dikirim ke DagsHub dengan Python 3.10.12")

if __name__ == "__main__":
    train_and_log()