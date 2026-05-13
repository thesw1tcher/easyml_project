from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import TYPE_CHECKING

from domain.project import Project, ProjectStatus
from ml.trainer import train_model, save_model, get_feature_importance

if TYPE_CHECKING:
    from domain.repository import ProjectRepository


def render_model_training(repo: ProjectRepository, project: Project, df: pd.DataFrame) -> None:
    """
    Рендер секции с тренировкой модели
    """
    st.divider()
    st.header("Model Training")

    if project.status != ProjectStatus.READY_FOR_ML:
        st.warning("Please configure target and features in the 'ML Configuration' section first.")
        return

    st.info(f"Task: {project.task_type}")
    
    col1, col2 = st.columns(2)
    with col1:
        algorithm = st.selectbox(
            "Select Algorithm",
            options=["CatBoost", "RandomForest", "Linear/Logistic Regression"]
        )
        test_size = st.slider("Test set size", 0.1, 0.5, 0.2, 0.05)
    
    with col2:
        iterations = st.number_input(
            "Iterations / Estimators", 
            min_value=10, 
            max_value=2000, 
            value=100, 
            step=50
        )
        tune_params = st.checkbox("Optimize hyperparameters (Grid Search)", value=False)

    if st.button("Train Model", type="primary", width="stretch"):
        with st.spinner(f"Training {algorithm} model..."):
            try:
                model, metrics = train_model(
                    df=df,
                    target=project.target_column,
                    features=project.feature_columns,
                    task_type=project.task_type,
                    algorithm=algorithm,
                    iterations=int(iterations),
                    test_size=test_size,
                    tune_hyperparams=tune_params
                )
                
                # Сохранение
                model_path = repo.model_path(project.project_id)
                save_model(model, model_path)
                
                st.success("Model trained successfully!")
                
                # Метрики
                st.subheader("Evaluation Metrics")
                m_cols = st.columns(len(metrics))
                for i, (m_name, m_val) in enumerate(metrics.items()):
                    m_cols[i].metric(m_name, f"{m_val:.4f}")
                
                # Важность признаков
                importance_df = get_feature_importance(model, project.feature_columns)
                if not importance_df.empty:
                    st.subheader("Feature Importance")
                    st.bar_chart(importance_df.set_index("Feature"))
                
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)
