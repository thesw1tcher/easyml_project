from __future__ import annotations

import joblib
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from typing import Any, Tuple, Optional

from domain.project import TaskType


from sklearn.model_selection import GridSearchCV, train_test_split
from sklearn.ensemble import RandomForestClassifier, RandomForestRegressor
from sklearn.linear_model import LogisticRegression, LinearRegression
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.impute import SimpleImputer
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import accuracy_score, f1_score, precision_score, recall_score, mean_squared_error, r2_score
import numpy as np

def train_model(
    df: pd.DataFrame, 
    target: str, 
    features: list[str], 
    task_type: TaskType,
    algorithm: str = "CatBoost",
    iterations: int = 100,
    test_size: float = 0.2,
    tune_hyperparams: bool = False
) -> Tuple[Any, dict, Tuple[np.ndarray, np.ndarray]]:
    """
    Обучение выбранной модели с препроцессингом и оценкой.
    Возвращает (model, metrics, (y_test, y_pred)).
    """
    X = df[features]
    y = df[target]

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=test_size, random_state=42)

    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()
    num_features = X.select_dtypes(exclude=['object', 'category']).columns.tolist()

    if algorithm == "CatBoost":
        if task_type == TaskType.CLASSIFICATION:
            model = CatBoostClassifier(iterations=iterations, verbose=False, allow_writing_files=False, cat_features=cat_features)
        else:
            model = CatBoostRegressor(iterations=iterations, verbose=False, allow_writing_files=False, cat_features=cat_features)
        
        if tune_hyperparams:
            param_grid = {
                'depth': [4, 6, 8],
                'learning_rate': [0.05, 0.1]
            }
            model.grid_search(
                param_grid, 
                X=X_train, 
                y=y_train, 
                cv=3, 
                partition_random_seed=42,
                verbose=False
            )
        else:
            model.fit(X_train, y_train, cat_features=cat_features)
        
    else:

        num_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='median')),
            ('scaler', StandardScaler())
        ])
        cat_transformer = Pipeline(steps=[
            ('imputer', SimpleImputer(strategy='most_frequent')),
            ('onehot', OneHotEncoder(handle_unknown='ignore', sparse_output=False))
        ])
        preprocessor = ColumnTransformer(transformers=[
            ('num', num_transformer, num_features),
            ('cat', cat_transformer, cat_features)
        ])

        if algorithm == "RandomForest":
            base = RandomForestClassifier(n_estimators=iterations) if task_type == TaskType.CLASSIFICATION else RandomForestRegressor(n_estimators=iterations)
        else:
            base = LogisticRegression(max_iter=1000) if task_type == TaskType.CLASSIFICATION else LinearRegression()

        model = Pipeline(steps=[('preprocessor', preprocessor), ('model', base)])
        
        if tune_hyperparams and algorithm == "RandomForest":
            param_grid = {'model__max_depth': [None, 10, 20]}
            grid = GridSearchCV(model, param_grid, cv=3, n_jobs=-1)
            grid.fit(X_train, y_train)
            model = grid.best_estimator_
        else:
            model.fit(X_train, y_train)

    y_pred = model.predict(X_test)
    metrics = evaluate_model(y_test, y_pred, task_type)
    return model, metrics, (y_test, y_pred)

def evaluate_model(y_test: pd.Series, y_pred: np.ndarray, task_type: TaskType) -> dict:
    if task_type == TaskType.CLASSIFICATION:
        return {
            "Accuracy": accuracy_score(y_test, y_pred),
            "Precision": precision_score(y_test, y_pred, average='weighted', zero_division=0),
            "Recall": recall_score(y_test, y_pred, average='weighted', zero_division=0),
            "F1-Score": f1_score(y_test, y_pred, average='weighted', zero_division=0)
        }
    else:
        return {
            "RMSE": np.sqrt(mean_squared_error(y_test, y_pred)),
            "R2-Score": r2_score(y_test, y_pred)
        }

def get_feature_importance(model: Any, features: list[str]) -> pd.DataFrame:
    """Извлекает важность признаков с поддержкой имен из Pipeline"""
    try:
        if hasattr(model, "get_feature_importance"): # CatBoost
            importances = model.get_feature_importance()
            feature_names = features
        elif hasattr(model, "steps"): # Scikit-learn Pipeline
            preprocessor = model.named_steps['preprocessor']
            try:
                feature_names = preprocessor.get_feature_names_out()
            except:
                feature_names = features
            
            base_model = model.steps[-1][1]
            if hasattr(base_model, "feature_importances_"):
                importances = base_model.feature_importances_
            elif hasattr(base_model, "coef_"):
                importances = np.abs(base_model.coef_)
                if len(importances.shape) > 1:
                    importances = np.mean(importances, axis=0)
            else:
                return pd.DataFrame()
        else:
            return pd.DataFrame()
        
        if len(importances) != len(feature_names):
            return pd.DataFrame()
            
        return pd.DataFrame({"Feature": feature_names, "Importance": importances}).sort_values(by="Importance", ascending=False)
    except:
        return pd.DataFrame()


def save_model(model: Any, path: str) -> None:
    """
    Save the trained model to the specified path.
    """
    joblib.dump(model, path)
