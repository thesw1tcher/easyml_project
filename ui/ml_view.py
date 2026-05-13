from __future__ import annotations

import pandas as pd
import streamlit as st
from typing import TYPE_CHECKING

from domain.project import Project, ProjectStatus
from ml.trainer import train_model, save_model

if TYPE_CHECKING:
    from domain.repository import ProjectRepository


def render_model_training(repo: ProjectRepository, project: Project, df: pd.DataFrame) -> None:
    """
    Renders the model training section.
    """
    st.divider()
    st.header("Model Training")

    if project.status != ProjectStatus.READY_FOR_ML:
        st.warning("Please configure target and features in the 'ML Configuration' section first.")
        return

    st.info(f"Task: {project.task_type}")
    tune_params = st.checkbox(
        "Optimize hyperparameters (Grid Search)", 
        value=False,
        help="This will try different combinations of parameters to find the best model, but will take more time."
    )

    iterations = st.number_input(
        "Number of iterations (trees)", 
        min_value=10, 
        max_value=2000, 
        value=100, 
        step=50,
        help="The number of trees in the ensemble. More trees can lead to better performance but will take longer to train."
    )

    if st.button("Train Model", type="primary"):
        with st.spinner("Training model..."):
            try:
                model = train_model(
                    df=df,
                    target=project.target_column,
                    features=project.feature_columns,
                    task_type=project.task_type,
                    iterations=int(iterations),
                    tune_hyperparams=tune_params
                )
                
                model_path = repo.model_path(project.project_id)
                save_model(model, model_path)
                
                st.success(f"Model successfully trained and saved to {model_path}")
            except Exception as e:
                st.error(f"Training failed: {e}")
