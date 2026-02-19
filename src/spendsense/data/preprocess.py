from pathlib import Path
import pandas as pd

def apply_category_mapping(df: pd.DataFrame, mapping_path: Path) -> pd.DataFrame:
    m = pd.read_csv(mapping_path)
    mapping = dict(zip(m["raw_category"].astype(str).str.strip(),
                       m["canonical_category"].astype(str).str.strip()))

    df = df.copy()
    df["category_raw"] = df["category"]  # keep original for audit
    df["category"] = df["category"].astype(str).str.strip().map(mapping).fillna(df["category"])
    return df