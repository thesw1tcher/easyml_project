from __future__ import annotations
import streamlit as st
from utils.config_loader import config
from domain.project import Project, ProjectStatus, TaskType
from domain.repository import ProjectRepository

def render_ml_config(repo: ProjectRepository, project: Project) -> None:
    """
    Изменение конфигурации модели (регрессия/классификация, выбор колонок для обучения и предиктов)
    """
    st.subheader("ML Configuration")
    
    target_col = st.selectbox(
        "Select Target Column",
        options=project.column_names,
        index=project.column_names.index(
            project.target_column) if project.target_column in project.column_names else 0
    )

    task_types = config["ml"]["task_types"]
    task_type = st.selectbox(
        "Select Task Type",
        options=[t.value for t in TaskType],
        index=[t.value for t in TaskType].index(project.task_type) if project.task_type in TaskType else 0
    )

    available_features = [
        c for c in project.column_names if c != target_col]
    default_features = [
        f for f in project.feature_columns if f in available_features]
    if not default_features:
        default_features = available_features

    feature_cols = st.multiselect(
        "Select Feature Columns",
        options=available_features,
        default=default_features
    )

    if st.button("Save ML Configuration", type="primary"):
        project.target_column = target_col
        project.task_type = TaskType(task_type)
        project.feature_columns = feature_cols
        project.status = ProjectStatus.READY_FOR_ML
        repo.save(project)
        st.success("ML Configuration saved.")
        st.rerun()
