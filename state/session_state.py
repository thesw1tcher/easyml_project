from __future__ import annotations

import streamlit as st


def init_state() -> None:
    """
    Streamlit по умолчанию забывает всё при каждом действии пользователя.
    Этот модуль инициализирует и управляет st.session_state, чтобы программа 
    помнила, какой проект сейчас открыт и какой DataFrame загружен в память.
    """
    defaults = {
        "active_project_id": None,
        "active_project": None,
        "active_df": None,
        "upload_error": None,
    }
    for key, value in defaults.items():
        if key not in st.session_state:
            st.session_state[key] = value


def set_active_project(project) -> None:
    st.session_state.active_project_id = project.project_id
    st.session_state.active_project = project


def clear_active_data() -> None:
    st.session_state.active_df = None
    st.session_state.upload_error = None
