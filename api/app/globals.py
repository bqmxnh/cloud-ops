import threading
from collections import deque
from river import metrics, drift

# ======================================================
# SHARED GLOBAL OBJECTS
# ======================================================

model = None
scaler = None
encoder = None
FEATURE_ORDER = None

model_lock = threading.Lock()

# PREDICTION HISTORY
prediction_history = {}

# ADWIN DRIFT DETECTOR
adwin = drift.ADWIN(delta=0.002)
error_buffer = deque(maxlen=50)

# DRIFT TIMELINE (list)
drift_timeline = []

# METRICS
metric_acc   = metrics.Accuracy()
metric_prec  = metrics.Precision()
metric_rec   = metrics.Recall()
metric_f1    = metrics.F1()
metric_kappa = metrics.CohenKappa()
metric_cm    = metrics.ConfusionMatrix()
