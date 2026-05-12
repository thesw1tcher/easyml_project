from __future__ import annotations
import pandas as pd
import streamlit as st
from utils.config_loader import config

def render_data_processing(repo, project, df: pd.DataFrame) -> None:
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
                            # Try to convert to float or int if possible
                            val = float(fill_value) if "." in fill_value else int(fill_value)
                        except ValueError:
                            # If conversion fails, check if column is string/object
                            if new_df[col].dtype == "object":
                                val = fill_value
                            else:
                                st.error(f"Cannot assign value '{fill_value}' to numeric column {col}")
                                return
                            
                        new_df[col] = new_df[col].fillna(val)

                st.session_state.active_df = new_df
                repo.save(project, df=new_df)
                st.success("Missing values filled and dataset updated.")
                st.rerun()
