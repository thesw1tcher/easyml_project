from __future__ import annotations
import pandas as pd
import streamlit as st
import matplotlib.pyplot as plt
import seaborn as sns

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
