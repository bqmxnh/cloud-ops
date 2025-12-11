from datetime import datetime
from app import globals as G

def add_drift_event(recent_err, global_err, reason):
    event = {
        "timestamp": datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f"),
        "recent_error": recent_err,
        "global_error": global_err,
        "reason": reason
    }
    G.drift_timeline.append(event)
    return event
