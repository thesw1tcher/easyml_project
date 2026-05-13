from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import TYPE_CHECKING

from domain.project import Project, ProjectStatus, TaskType
from ml.trainer import train_model, save_model, get_feature_importance

if TYPE_CHECKING:
    from domain.repository import ProjectRepository


import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix

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
                # Теперь возвращается 3 значения: модель, метрики и (y_test, y_pred)
                model, metrics, eval_data = train_model(
                    df=df,
                    target=project.target_column,
                    features=project.feature_columns,
                    task_type=project.task_type,
                    algorithm=algorithm,
                    iterations=int(iterations),
                    test_size=test_size,
                    tune_hyperparams=tune_params
                )
                
                # Сохранение модели
                model_path = repo.model_path(project.project_id)
                save_model(model, model_path)
                
                st.success("Model trained successfully!")
                
                # Разделение на вкладки для анализа
                tab_metrics, tab_importance, tab_advanced = st.tabs([
                    "📊 Metrics", "🔍 Feature Importance", "📈 Advanced Analysis"
                ])
                
                with tab_metrics:
                    st.subheader("Evaluation Metrics")
                    m_cols = st.columns(len(metrics))
                    for i, (m_name, m_val) in enumerate(metrics.items()):
                        m_cols[i].metric(m_name, f"{m_val:.4f}")
                    
                    st.info("Metrics are calculated on a hidden test set (unseen data).")
                
                with tab_importance:
                    importance_df = get_feature_importance(model, project.feature_columns)
                    if not importance_df.empty:
                        st.subheader("Feature Importance")
                        st.bar_chart(importance_df.set_index("Feature"))
                    else:
                        st.warning("Feature importance is not available for this model/configuration.")
                
                with tab_advanced:
                    y_test, y_pred = eval_data
                    if project.task_type == TaskType.CLASSIFICATION:
                        st.subheader("Confusion Matrix")
                        cm = confusion_matrix(y_test, y_pred)
                        fig, ax = plt.subplots()
                        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', ax=ax)
                        ax.set_xlabel('Predicted')
                        ax.set_ylabel('Actual')
                        st.pyplot(fig)
                    else:
                        st.subheader("Residuals Plot")
                        residuals = y_test - y_pred
                        fig, ax = plt.subplots()
                        ax.scatter(y_pred, residuals, alpha=0.5)
                        ax.axhline(y=0, color='r', linestyle='--')
                        ax.set_xlabel('Predicted values')
                        ax.set_ylabel('Residuals')
                        st.pyplot(fig)
                
            except Exception as e:
                st.error(f"Training failed: {e}")
                st.exception(e)
