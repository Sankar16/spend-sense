from pathlib import Path
import pandas as pd

from src.spendsense.data.preprocess import apply_category_mapping

def main():
    processed_dir = Path("data/processed")
    processed_dir.mkdir(parents=True, exist_ok=True)

    # Load your current canonical dataset
    in_path = processed_dir / "transactions_clean.parquet"
    if not in_path.exists():
        raise FileNotFoundError("transactions_clean.parquet not found. Generate it first.")

    df = pd.read_parquet(in_path)

    # Apply category mapping
    mapping_path = Path("data/raw/category_mapping.csv")
    if not mapping_path.exists():
        raise FileNotFoundError(f"{mapping_path} not found. Create it first.")

    df2 = apply_category_mapping(df, mapping_path)

    # Save back (overwrite)
    df2.to_parquet(in_path, index=False)
    print("Updated canonical dataset with standardized categories:", in_path)

if __name__ == "__main__":
    main()