from __future__ import annotations
import pandas as pd
import streamlit as st
import os
from typing import TYPE_CHECKING

from domain.project import Project, ProjectStatus
from ml.inference import load_model, make_prediction, validate_features

if TYPE_CHECKING:
    from domain.repository import ProjectRepository

def render_inference(repo: ProjectRepository, project: Project) -> None:
    """
    Рендер страницы предсказаний
    """
    st.header(f"Model Inference: {project.name}")
    
    model_path = repo.model_path(project.project_id)
    if not os.path.exists(model_path):
        st.warning("No trained model found for this project. Please train a model first.")
        return

    tab_batch, tab_single = st.tabs(["📁 Batch Prediction", "✏️ Single Prediction"])

    with tab_batch:
        render_batch_inference(project, model_path)
    
    with tab_single:
        render_single_inference(project, model_path)

def render_batch_inference(project: Project, model_path: str) -> None:
    st.subheader("Upload Data for Prediction")
    uploaded_file = st.file_uploader("Choose a CSV file", type="csv")
    
    if uploaded_file:
        try:
            input_df = pd.read_csv(uploaded_file)
            st.write("Preview of uploaded data:", input_df.head())
            
            success, missing = validate_features(input_df, project.feature_columns)
            if not success:
                st.error(f"Missing required features: {', '.join(missing)}")
                return
            
            if st.button("Generate Predictions"):
                with st.spinner("Processing..."):
                    model = load_model(model_path)
                    # Используем только нужные фичи
                    X = input_df[project.feature_columns]
                    results = make_prediction(model, X)
                    
                    st.success("Predictions generated!")
                    st.write(results.head())
                    
                    csv = results.to_csv(index=False).encode('utf-8')
                    st.download_button(
                        "Download Results as CSV",
                        csv,
                        "predictions.csv",
                        "text/csv"
                    )
        except Exception as e:
            st.error(f"Error processing file: {e}")

def render_single_inference(project: Project, model_path: str) -> None:
    st.subheader("Interactive Prediction Form")
    st.write("Enter values for a single prediction:")
    
    input_data = {}
    cols = st.columns(2)
    for i, feature in enumerate(project.feature_columns):
        with cols[i % 2]:
            input_data[feature] = st.text_input(f"Enter {feature}", key=f"single_{feature}")

    if st.button("Predict"):
        try:
            single_df = pd.DataFrame([input_data])
            
            for col in single_df.columns:
                try:
                    single_df[col] = pd.to_numeric(single_df[col])
                except:
                    pass
            
            model = load_model(model_path)
            results = make_prediction(model, single_df)
            
            prediction = results['prediction'].iloc[0]
            
            st.metric("Model Prediction", f"{prediction}")
            
            if 'probability' in results.columns:
                st.write(f"Confidence: {results['probability'].iloc[0]:.2%}")
                
        except Exception as e:
            st.error(f"Prediction error: {e}")
