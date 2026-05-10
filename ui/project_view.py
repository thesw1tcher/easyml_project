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


def render_project_header(project: Project) -> None:
    """
    Заголовок проекта, отображает общую информацию
    """
    st.title(project.name)
    cols = st.columns(4)
    cols[0].metric("Rows", project.n_rows)
    cols[1].metric("Columns", project.n_cols)
    cols[2].metric("Status", project.status)
    cols[3].metric("Updated", project.updated_at[:19].replace("T", " "))


def render_project_actions(repo, project: Project, df: Optional[pd.DataFrame]) -> None:
    """
    Сохранение/сброс проекта.
    """
    left, right = st.columns([1, 1])
    with left:
        if st.button("Save project", type="primary", use_container_width=True):
            repo.save(project, df=df if isinstance(df, pd.DataFrame) else None)
            st.success("Project saved.")
            st.rerun()
    with right:
        if st.button("Clear loaded table", use_container_width=True):
            st.session_state.active_df = None
            project.dataset_filename = None
            project.dataset_snapshot_path = None
            project.n_rows = 0
            project.n_cols = 0
            project.column_names = []
            project.status = "empty"
            repo.save(project)
            st.rerun()


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


def render_dataset_preview(df: pd.DataFrame) -> None:
    """
    Рендер трёх вкладок:
    Вывод загруженной таблицы через streamlit.dataframe
    Вывод информации о колонках
    Вывод информации о NaN'ах
    """
    st.subheader("Preview")
    tab1, tab2, tab3 = st.tabs(["Table", "Info", "Missing values"])

    with tab1:
        st.dataframe(df.head(200), width="stretch")

    with tab2:
        st.write("**Dtypes:**")
        st.dataframe(pd.DataFrame(df.dtypes.astype(str), columns=[
                     "dtype"]), use_container_width=True)

    with tab3:
        missing = df.isna().sum().sort_values(ascending=False)
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values found.")
        else:
            st.dataframe(missing.to_frame("missing_count"),
                         use_container_width=True)


def render_data_processing(repo, project: Project, df: pd.DataFrame) -> None:
    """
    Заполнение NaN'ов медианой, средним или заданным значением + 
    изменение конфигурации модели (регрессия/классификация, выбор колонок для обучения и предиктов)
    """
    st.divider()
    st.header("Data Processing")

    col1, col2 = st.columns(2)

    with col1:
        st.subheader("Fill Missing Values")
        cols_with_nan = df.columns[df.isna().any()].tolist()
        if not cols_with_nan:
            st.success("No missing values detected.")
        else:
            selected_cols = st.multiselect(
                "Select columns to fill", cols_with_nan)
            method = st.selectbox(
                "Fill method", ["Mean", "Median", "Constant"])

            fill_value = None
            if method == "Constant":
                fill_value = st.text_input("Enter constant value", value="0")

            if st.button("Apply Fill NaN", type="primary"):
                if not selected_cols:
                    st.warning("Please select at least one column.")
                else:
                    new_df = df.copy()
                    for col in selected_cols:
                        if method == "Mean":
                            if pd.api.types.is_numeric_dtype(new_df[col]):
                                new_df[col] = new_df[col].fillna(
                                    new_df[col].mean())
                            else:
                                st.error(
                                    f"Cannot apply Mean to non-numeric column: {col}")
                                return
                        elif method == "Median":
                            if pd.api.types.is_numeric_dtype(new_df[col]):
                                new_df[col] = new_df[col].fillna(
                                    new_df[col].median())
                            else:
                                st.error(
                                    f"Cannot apply Median to non-numeric column: {col}")
                                return
                        elif method == "Constant":
                            try:
                                val = float(fill_value) if "." in fill_value else int(
                                    fill_value)
                            except ValueError:
                                if new_df[col].dtype == "str":
                                    val = fill_value
                                else:
                                    raise ValueError("Can't assign non-string column with string values")
                                
                            new_df[col] = new_df[col].fillna(val)

                    st.session_state.active_df = new_df
                    repo.save(project, df=new_df)
                    st.success("Missing values filled and dataset updated.")
                    st.rerun()

    with col2:
        st.subheader("ML Configuration")
        target_col = st.selectbox(
            "Select Target Column",
            options=project.column_names,
            index=project.column_names.index(
                project.target_column) if project.target_column in project.column_names else 0
        )

        task_type = st.selectbox(
            "Select Task Type",
            options=["classification", "regression"],
            index=0 if project.task_type == "classification" else 1 if project.task_type == "regression" else 0
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
            project.task_type = task_type
            project.feature_columns = feature_cols
            project.status = "ready_for_ml"
            repo.save(project)
            st.success("ML Configuration saved.")
            st.rerun()


def render_eda(df: pd.DataFrame) -> None:
    """
    Вывод коробок с усами для числовых признаков и столбчатых диаграмм для категориальных + корреляции.
    """
    st.divider()
    st.header("Exploratory Data Analysis")

    num_cols = df.select_dtypes(include=["number"]).columns.tolist()
    cat_cols = df.select_dtypes(exclude=["number"]).columns.tolist()

    tab1, tab2, tab3 = st.tabs(["Numerical (Box Plots)", "Categorical (Bar Charts)", "Correlations (Heatmap)"])

    with tab1:
        if num_cols:
            col_to_plot = st.selectbox(
                "Select column for Box Plot", num_cols, key="box_plot_col")
            if st.button("Show Box Plot"):
                st.vega_lite_chart(df, {
                    'mark': {'type': 'boxplot', 'extent': 'min-max'},
                    'encoding': {
                        'y': {'field': col_to_plot, 'type': 'quantitative'}
                    }
                }, use_container_width=True)
        else:
            st.info("No numerical columns found.")

    with tab2:
        if cat_cols:
            col_to_plot = st.selectbox(
                "Select column for Bar Chart", cat_cols, key="bar_chart_col")
            if st.button("Show Bar Chart"):
                counts = df[col_to_plot].value_counts().reset_index()
                counts.columns = [col_to_plot, "count"]
                st.bar_chart(counts, x=col_to_plot, y="count")
        else:
            st.info("No categorical columns found.")

    with tab3:
        if num_cols and len(num_cols) > 1:
            fig, ax = plt.subplots(figsize=(10, 8))
            sns.heatmap(df[num_cols].corr(), annot=True, cmap='coolwarm', fmt=".2f", ax=ax)
            st.pyplot(fig)
        else:
            st.info("Need at least two numerical columns for correlation heatmap.")
