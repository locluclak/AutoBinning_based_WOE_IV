from pathlib import Path

import pandas as pd


def load_data(file_path: str) -> pd.DataFrame:
    path = Path(file_path)

    match path.suffix.lower():
        case ".csv":
            return pd.read_csv(file_path)
        case ".parquet":
            return pd.read_parquet(file_path)
        case ".xlsx" | ".xls":
            return pd.read_excel(file_path)
        case ".json":
            return pd.read_json(file_path)
        case _:
            raise ValueError(f"Định dạng file không được hỗ trợ: {path.suffix}")


def write_data(df: pd.DataFrame, file_path: str) -> str:
    path = Path(file_path)
    match path.suffix.lower():
        case ".csv":
            df.to_csv(file_path, index=False, encoding="utf-8-sig")
        case ".parquet":
            df.to_parquet(file_path, index=False)
        case _:
            raise ValueError(f"Định dạng file không được hỗ trợ: {path.suffix}")
    return str(path)