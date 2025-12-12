import threading
from collections import deque
from river import metrics, drift
import asyncio
from collections import defaultdict

# ======================================================
# SHARED GLOBAL OBJECTS
# ======================================================

model = None
scaler = None
encoder = None
FEATURE_ORDER = None

model_lock = threading.Lock()

current_model_version = None
model_reload_count = 0

# PREDICTION HISTORY
prediction_history = {}

# ADWIN DRIFT DETECTOR
adwin = drift.ADWIN(delta=0.1)
error_buffer = deque(maxlen=50)

# DRIFT TIMELINE (list)
drift_timeline = []

prediction_events = {} 

# METRICS
metric_acc   = metrics.Accuracy()
metric_prec  = metrics.Precision()
metric_rec   = metrics.Recall()
metric_f1    = metrics.F1()
metric_kappa = metrics.CohenKappa()
metric_cm    = metrics.ConfusionMatrix()

async def cleanup_old_events():
    """Remove events older than 5 minutes"""
    while True:
        await asyncio.sleep(300)  # Every 5 mins
        current_keys = list(prediction_events.keys())
        for flow_id in current_keys:
            if flow_id not in prediction_history:
                prediction_events.pop(flow_id, None)