from __future__ import annotations

import joblib
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from typing import Any, Tuple, Optional

from domain.project import TaskType


from sklearn.model_selection import GridSearchCV

def train_model(
    df: pd.DataFrame, 
    target: str, 
    features: list[str], 
    task_type: TaskType,
    iterations: int = 100,
    tune_hyperparams: bool = False
) -> Any:
    """
    Обучение модели на основании типа задачи с опциональным поиском гиперпараметров.
    """
    X = df[features]
    y = df[target]

    cat_features = X.select_dtypes(include=['object', 'category']).columns.tolist()

    if task_type == TaskType.CLASSIFICATION:
        base_model = CatBoostClassifier(
            iterations=iterations,
            verbose=False,
            allow_writing_files=False,
            cat_features=cat_features
        )
    else:
        base_model = CatBoostRegressor(
            iterations=iterations,
            verbose=False,
            allow_writing_files=False,
            cat_features=cat_features
        )

    if tune_hyperparams:
        param_grid = {
            'iterations': [iterations, iterations * 2],
            'depth': [4, 6, 8],
            'learning_rate': [0.01, 0.05, 0.1],
        }
        grid_search = GridSearchCV(
            estimator=base_model,
            param_grid=param_grid,
            cv=3,
            scoring='accuracy' if task_type == TaskType.CLASSIFICATION else 'neg_mean_squared_error',
            n_jobs=-1
        )
        grid_search.fit(X, y)
        return grid_search.best_estimator_
    else:
        base_model.fit(X, y, cat_features=cat_features)
        return base_model


def save_model(model: Any, path: str) -> None:
    """
    Save the trained model to the specified path.
    """
    joblib.dump(model, path)
