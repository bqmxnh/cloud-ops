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

prediction_events = {} 

# METRICS
metric_acc   = metrics.Accuracy()
metric_prec  = metrics.Precision()
metric_rec   = metrics.Recall()
metric_f1    = metrics.F1()
metric_kappa = metrics.CohenKappa()
metric_cm    = metrics.ConfusionMatrix()
