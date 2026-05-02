from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import streamlit as st

from repositories.local_project_repository import LocalProjectRepository
from state.session_state import init_state
from ui.project_view import (
    render_dataset_preview,
    render_empty_state,
    render_project_actions,
    render_project_header,
    render_upload_block,
)
from ui.sidebar import sidebar_existing_projects, sidebar_new_project


APP_TITLE = "EasyML"


def main() -> None:
    st.set_page_config(page_title=APP_TITLE, layout="wide")
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
    except Exception:
        pass

    render_project_header(project)
    render_project_actions(repo, project, st.session_state.active_df)
    render_upload_block(repo, project)

    df = st.session_state.active_df
    if isinstance(df, pd.DataFrame):
        render_dataset_preview(df)
    else:
        st.subheader("Preview")
        st.caption("Upload a CSV file to see the table preview here.")

    st.divider()
    with st.expander("Project JSON preview"):
        st.code(json.dumps(project.to_dict(),
                ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
