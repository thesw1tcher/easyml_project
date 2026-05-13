from __future__ import annotations

from pathlib import Path

import pandas as pd
import streamlit as st

from domain.project import Project
from domain.repository import ProjectRepository
from state.session_state import clear_active_data, set_active_project


def sidebar_new_project(repo: ProjectRepository) -> None:
    st.sidebar.header("Project")
    with st.sidebar.form("create_project_form", clear_on_submit=True):
        name = st.text_input(
            "Project name", placeholder="My first EasyML project")
        create_clicked = st.form_submit_button("Create project")

    if create_clicked:
        if not name.strip():
            st.sidebar.error("Enter a project name.")
        else:
            project = Project.create(name)
            repo.save(project)
            set_active_project(project)
            clear_active_data()
            st.rerun()


def sidebar_existing_projects(repo: ProjectRepository) -> None:
    projects = repo.list_projects()
    st.sidebar.subheader("Open existing")

    if not projects:
        st.sidebar.caption("No saved projects yet.")
        return

    options = {
        f"{p.name}  ·  {p.updated_at[:19].replace('T', ' ')}": p for p in projects}
    selected_label = st.sidebar.selectbox(
        "Recent projects", list(options.keys()))
    selected_project = options[selected_label]

    col_a, col_b = st.sidebar.columns(2)
    with col_a:
        if st.button("Load", use_container_width=True):
            set_active_project(selected_project)
            clear_active_data()
            if selected_project.dataset_snapshot_path:
                snapshot = Path(selected_project.dataset_snapshot_path)
                if snapshot.exists():
                    st.session_state.active_df = pd.read_csv(snapshot)
            st.rerun()
    with col_b:
        if st.button("Delete", use_container_width=True):
            repo.delete(selected_project.project_id)
            if st.session_state.active_project_id == selected_project.project_id:
                st.session_state.active_project_id = None
                st.session_state.active_project = None
                st.session_state.active_df = None
            st.rerun()
