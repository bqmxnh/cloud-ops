def sanitize(features: dict):
    clean = {}
    for k, v in features.items():
        try:
            clean[k] = float(v)
        except:
            clean[k] = 0.0
    return clean
