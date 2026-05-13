import joblib
import pandas as pd
from typing import Any, Optional, Tuple

def load_model(model_path: str) -> Any:
    """Загрузка обученной модели"""
    return joblib.load(model_path)

def make_prediction(model: Any, input_df: pd.DataFrame) -> pd.DataFrame:
    """
    Применяет модель к входным данным и возвращает результат в виде DataFrame.
    """
    predictions = model.predict(input_df)
    
    result_df = input_df.copy()
    result_df['prediction'] = predictions
    
    if hasattr(model, "predict_proba"):
        try:
            probas = model.predict_proba(input_df)
            if probas.shape[1] == 2: # бинарная классификация
                result_df['probability'] = probas[:, 1]
        except:
            pass
            
    return result_df

def validate_features(input_df: pd.DataFrame, required_features: list[str]) -> Tuple[bool, list[str]]:
    """
    Проверяет, все ли необходимые признаки присутствуют в данных.
    Возвращает (успех, список отсутствующих колонок).
    """
    missing = [f for f in required_features if f not in input_df.columns]
    return len(missing) == 0, missing
