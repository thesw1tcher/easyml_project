from __future__ import annotations

import json
import pandas as pd
import streamlit as st

from repositories.local_project_repository import LocalProjectRepository
from state.session_state import init_state
from ui.project_view import (
    render_dataset_preview,
    render_empty_state,
    render_project_actions,
    render_project_header,
    render_upload_block
)
from domain.exceptions import ProjectNotFoundError
from ui.eda_view import render_eda
from ui.processing_view import render_data_processing
from ui.ml_config_view import render_ml_config
from ui.ml_view import render_model_training
from ui.sidebar import sidebar_existing_projects, sidebar_new_project
from utils.config_loader import config


def main() -> None:
    st.set_page_config(page_title=config["app"]["title"], layout="wide")
    init_state()
    repo = LocalProjectRepository()

    sidebar_new_project(repo)
    sidebar_existing_projects(repo)

    project = st.session_state.active_project

    if project is None:
        render_empty_state()
        return

    try:
        project = repo.load(project.project_id)
        st.session_state.active_project = project
    except (ProjectNotFoundError, json.JSONDecodeError) as e:
        st.error(f"Failed to load active project: {e}")
        st.session_state.active_project = None
        st.session_state.active_project_id = None
        st.rerun()
    except Exception as e:
        st.error(f"An unexpected error occurred while loading project: {e}")
        st.session_state.active_project = None
        st.session_state.active_project_id = None
        st.rerun()

    render_project_header(project)
    render_project_actions(repo, project, st.session_state.active_df)
    render_upload_block(repo, project)

    df = st.session_state.active_df
    if isinstance(df, pd.DataFrame):
        render_dataset_preview(df)
        render_eda(df)
        
        col1, col2 = st.columns(2)
        with col1:
            render_data_processing(repo, project, df)
        with col2:
            st.divider()
            st.header("ML Configuration")
            render_ml_config(repo, project)
            render_model_training(repo, project, df)
    else:
        st.subheader("Preview")
        st.caption("Upload a CSV file to see the table preview here.")

    st.divider()
    with st.expander("Project JSON preview"):
        st.code(json.dumps(project.to_dict(),
                ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
