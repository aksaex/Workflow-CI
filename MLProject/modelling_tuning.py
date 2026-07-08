import pandas as pd
import mlflow
import dagshub
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import GridSearchCV
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix
import matplotlib.pyplot as plt
import seaborn as sns
import os

# 1. Hubungkan ke DagsHub (GANTI DENGAN USERNAME DAN NAMA REPO DAGSHUB-MU)
dagshub.init(repo_owner='aksaex', repo_name='MSML_Proyek_Akhir', mlflow=True)

def train_and_log():
    # 2. Load dataset yang sudah di-preprocessing
    # Sesuaikan path-nya jika perlu
    train = pd.read_csv('heart_preprocessing/train.csv')
    test = pd.read_csv('heart_preprocessing/test.csv')
    
    X_train = train.drop('output', axis=1)
    y_train = train['output']
    X_test = test.drop('output', axis=1)
    y_test = test['output']

    # 3. Inisialisasi Model & Hyperparameter Tuning (GridSearch)
    rf = RandomForestClassifier(random_state=42)
    param_grid = {
        'n_estimators': [50, 100],
        'max_depth': [5, 10]
    }
    grid_search = GridSearchCV(rf, param_grid, cv=3)

    print("Mulai melatih model dan mencari parameter terbaik...")
    
    # 4. Memulai MLflow run dengan Manual Logging
    with mlflow.start_run():
        grid_search.fit(X_train, y_train)
        best_model = grid_search.best_estimator_
        y_pred = best_model.predict(X_test)

        # A. Manual Log Parameters
        mlflow.log_param("best_n_estimators", best_model.n_estimators)
        mlflow.log_param("best_max_depth", best_model.max_depth)

        # B. Manual Log Metrics
        mlflow.log_metric("accuracy", accuracy_score(y_test, y_pred))
        mlflow.log_metric("precision", precision_score(y_test, y_pred))
        mlflow.log_metric("recall", recall_score(y_test, y_pred))

        # C. Log Model Utama
        mlflow.sklearn.log_model(best_model, "random_forest_model")

        # D. Log Artefak Tambahan 1: Confusion Matrix
        cm = confusion_matrix(y_test, y_pred)
        plt.figure(figsize=(6,4))
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues')
        plt.title("Confusion Matrix")
        plt.savefig("confusion_matrix.png")
        mlflow.log_artifact("confusion_matrix.png")
        plt.close()

        # E. Log Artefak Tambahan 2: Feature Importance
        importances = best_model.feature_importances_
        plt.figure(figsize=(8,6))
        plt.barh(X_train.columns, importances)
        plt.title("Feature Importance")
        plt.savefig("feature_importance.png")
        mlflow.log_artifact("feature_importance.png")
        plt.close()

        print("Selesai! Model dan seluruh metrik/artefak berhasil dikirim ke DagsHub.")

if __name__ == "__main__":
    train_and_log()