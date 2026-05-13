from __future__ import annotations

import joblib
import pandas as pd
from catboost import CatBoostClassifier, CatBoostRegressor
from typing import Any, Tuple, Optional

from domain.project import TaskType


def train_model(
    df: pd.DataFrame, 
    target: str, 
    features: list[str], 
    task_type: TaskType
) -> Any:
    """
    Обучение модели на основании типа задачи
    """
    X = df[features]
    y = df[target]

    if task_type == TaskType.CLASSIFICATION:
        model = CatBoostClassifier(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            allow_writing_files=False
        )
    else:
        model = CatBoostRegressor(
            iterations=100,
            depth=6,
            learning_rate=0.1,
            verbose=False,
            allow_writing_files=False
        )

    model.fit(X, y)
    return model


def save_model(model: Any, path: str) -> None:
    """
    Save the trained model to the specified path.
    """
    joblib.dump(model, path)
