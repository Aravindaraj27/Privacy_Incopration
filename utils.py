import pandas as pd
import random

def hash_data(value):
    return hash(value)

def generalize_column(value):
    if isinstance(value, int):
        return value // 10 * 10
    elif isinstance(value, float):
        return round(value, -1)
    elif isinstance(value, str):
        return value[:3] + '...'
    return value

def mask_data(value):
    s = str(value)
    return s[:2] + '*' * (len(s) - 2) if len(s) > 2 else '*' * len(s)

def scramble_data(value):
    return ''.join(random.sample(str(value), len(str(value)))) if value else value

def anonymize_column(column: pd.Series, technique: str) -> pd.Series:
    technique_map = {
        "masking": lambda col: col.apply(mask_data),
        "pseudonymization": lambda col: col.apply(lambda x: f"pseudo_{hash(x)}"),
        "generalization": lambda col: col.apply(generalize_column),
        "suppression": lambda col: col.apply(lambda _: None),
        "synthetic": lambda col: col.apply(lambda x: f"synthetic_{random.randint(1000,9999)}"),
        "hashing": lambda col: col.apply(hash_data),
        "k_anonymity": lambda col: col,  # Placeholder for actual k-anonymity logic
        "data_swapping": lambda col: col.sample(frac=1).reset_index(drop=True),
        "scrambling": lambda col: col.apply(scramble_data),
        "shuffling": lambda col: col.sample(frac=1).reset_index(drop=True),
    }
    return technique_map.get(technique, lambda col: col)(column)

def detect_anonymization_technique(column: pd.Series) -> str:
    """
    Dummy detection logic: uses heuristics based on type and value characteristics.
    """
    if column.dtype == 'object':
        if column.str.contains('@').any():
            return "scrambling"
        elif column.str.len().mean() > 10:
            return "pseudonymization"
        else:
            return "masking"
    elif pd.api.types.is_numeric_dtype(column):
        return "generalization"
    return "suppression"
