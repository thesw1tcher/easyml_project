from __future__ import annotations
import pandas as pd
import streamlit as st
from utils.config_loader import config
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from domain.project import Project
    from domain.repository import ProjectRepository

def render_data_processing(repo: ProjectRepository, project: Project, df: pd.DataFrame) -> None:
    """
    Заполнение NaN'ов медианой, средним или заданным значением.
    """
    st.divider()
    st.header("Data Processing")

    st.subheader("Fill Missing Values")
    cols_with_nan = df.columns[df.isna().any()].tolist()
    if not cols_with_nan:
        st.success("No missing values detected.")
    else:
        selected_cols = st.multiselect(
            "Select columns to fill", cols_with_nan)
        
        methods = config["data_processing"]["fill_methods"]
        method = st.selectbox("Fill method", methods)

        fill_value = None
        if method == "Constant":
            fill_value = st.text_input("Enter constant value", value="0")

        if st.button("Apply Fill NaN", type="primary"):
            if not selected_cols:
                st.warning("Please select at least one column.")
            else:
                new_df = df.copy()
                for col in selected_cols:
                    try:
                        if method == "Mean":
                            if pd.api.types.is_numeric_dtype(new_df[col]):
                                new_df[col] = new_df[col].fillna(new_df[col].mean())
                            else:
                                st.error(f"Cannot apply Mean to non-numeric column: {col}")
                                continue
                        elif method == "Median":
                            if pd.api.types.is_numeric_dtype(new_df[col]):
                                new_df[col] = new_df[col].fillna(new_df[col].median())
                            else:
                                st.error(f"Cannot apply Median to non-numeric column: {col}")
                                continue
                        elif method == "Constant":
                            val = fill_value
                            try:
                                if "." in fill_value:
                                    val = float(fill_value)
                                else:
                                    val = int(fill_value)
                            except ValueError:
                                pass
                            
                            if not pd.api.types.is_object_dtype(new_df[col]):
                                try:
                                    pd.Series([val]).astype(new_df[col].dtype)
                                except (ValueError, TypeError):
                                    st.error(f"Value '{fill_value}' is incompatible with numeric column '{col}'")
                                    continue
                                
                            new_df[col] = new_df[col].fillna(val)
                    except Exception as e:
                        st.error(f"Error processing column '{col}': {e}")
                        continue

                st.session_state.active_df = new_df
                repo.save(project, df=new_df)
                st.success("Missing values filled and dataset updated.")
                st.rerun()
