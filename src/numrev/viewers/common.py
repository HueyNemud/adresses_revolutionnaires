"""Éléments communs aux deux viewers Streamlit."""

import pandas as pd

PAGE_SIZE_OPTIONS = [25, 50, 100, 200]


def text_mask(df: pd.DataFrame, query: str, columns: list[str], regex: bool = False) -> pd.Series:
    """Lignes dont l'une des `columns` contient `query` (sans casse)."""
    mask = pd.Series(False, index=df.index)
    for column in columns:
        mask |= df[column].astype(str).str.contains(query, case=False, na=False, regex=regex)
    return mask
