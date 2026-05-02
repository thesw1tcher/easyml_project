from __future__ import annotations

from typing import Optional

import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

from domain.project import Project
from utils.helpers import load_dataset


APP_TITLE = "EasyML"


def render_empty_state() -> None:
    """
    Первоначальное состояние до открытия проекта.
    """
    st.title(APP_TITLE)
    st.markdown(
        """
        Create a project or open an existing one, then load a table (CSV or Excel) file.
        """
    )

def render_upload_block(repo, project: Project) -> None:
    """
    Загрузка файла + выбор разделителя
    """
    st.subheader("Upload CSV or Excel")

    col1, col2 = st.columns([3, 1])

    with col2:
        delimiter = st.text_input(
            "Delimiter", value=",", help="Used for CSV files only.")

    with col1:
        uploaded = st.file_uploader(
            "Choose a file",
            type=["csv", "xlsx", "xls"],
            accept_multiple_files=False,
        )

    if uploaded is not None:
        try:
            kwargs = {}
            if uploaded.name.lower().endswith(".csv"):
                kwargs["sep"] = delimiter

            df = load_dataset(uploaded.getvalue(), uploaded.name, **kwargs)
            st.session_state.active_df = df
            project.dataset_filename = uploaded.name
            project.n_rows = int(df.shape[0])
            project.n_cols = int(df.shape[1])
            project.column_names = df.columns.astype(str).tolist()
            project.status = "dataset_loaded"
            repo.save(project, df=df)
            st.success(
                f"Loaded {uploaded.name}: {df.shape[0]} rows × {df.shape[1]} columns")
            st.session_state.upload_error = None
        except Exception as exc:
            st.session_state.upload_error = str(exc)
            st.error(f"Could not read file: {exc}")

    if st.session_state.upload_error:
        st.warning(st.session_state.upload_error)
