from __future__ import annotations

from io import BytesIO

import pandas as pd
import streamlit as st


@st.cache_data(show_spinner=True)
def load_dataset(file_bytes: bytes, filename: str, **kwargs) -> pd.DataFrame:
    """
    Loads a dataset from bytes based on the file extension.
    Supports CSV and Excel files.
    """
    buffer = BytesIO(file_bytes)
    ext = filename.split(".")[-1].lower()

    if ext == "csv":
        return pd.read_csv(buffer, **kwargs)
    elif ext in ["xlsx", "xls"]:
        return pd.read_excel(buffer, **kwargs)
    else:
        raise ValueError(f"Unsupported file extension: {ext}")
