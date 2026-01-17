import pandas as pd

def sanitize(features: dict):
    """
    Convert features dict to numeric values.
    Handles string numbers and invalid values.
    """
    clean = {}
    for k, v in features.items():
        try:
            # Try direct conversion first
            clean[k] = float(v)
        except (ValueError, TypeError):
            # Fall back to pandas' robust numeric conversion
            converted = pd.to_numeric(v, errors='coerce')
            # If conversion failed (NaN), use 0
            clean[k] = converted if pd.notna(converted) else 0.0
    return clean
