from __future__ import annotations

import json
from typing import Optional

import pandas as pd
import streamlit as st

from repositories.local_project_repository import LocalProjectRepository
from state.session_state import init_state
from ui.project_view import (
    render_empty_state,
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
    
    render_upload_block(repo, project)

    st.divider()
    with st.expander("Project JSON preview"):
        st.code(json.dumps(project.to_dict(),
                ensure_ascii=False, indent=2), language="json")


if __name__ == "__main__":
    main()
