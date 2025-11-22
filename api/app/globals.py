import threading
from river import metrics, drift

# ==========================
# SHARED GLOBAL OBJECTS
# ==========================

# Model components (được set khi load)
model = None
scaler = None
encoder = None

# Thread lock dùng chung
model_lock = threading.Lock()

# Prediction history dùng giữa predict → feedback
prediction_history = {}

# Metrics dùng chung
metric_acc = metrics.Accuracy()
metric_prec = metrics.Precision()
metric_rec = metrics.Recall()
metric_f1 = metrics.F1()
metric_kappa = metrics.CohenKappa()

# Drift detector
adwin = drift.ADWIN(delta=0.01)
