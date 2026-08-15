import json
import time

import numpy as np
import pandas as pd
from lightgbm import LGBMClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import (
    roc_auc_score,
    accuracy_score,
    f1_score,
    precision_score,
    recall_score,
)

DATA_PATH = "creditcard.csv"

# --- Load data ---
t0 = time.time()
df = pd.read_csv(DATA_PATH)
load_time = time.time() - t0

X = df.drop(columns=["Class"])
y = df["Class"]

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42, stratify=y
)

# --- Train ---
model = LGBMClassifier(
    n_estimators=500,
    learning_rate=0.05,
    num_leaves=31,
    random_state=42,
    n_jobs=-1,
)

t0 = time.time()
model.fit(
    X_train,
    y_train,
    eval_set=[(X_test, y_test)],
    eval_metric="auc",
    callbacks=[],
)
train_time = time.time() - t0

best_iteration = model.best_iteration_ or model.n_estimators_

# --- Evaluate ---
y_pred_proba = model.predict_proba(X_test)[:, 1]
y_pred = model.predict(X_test)

auc = roc_auc_score(y_test, y_pred_proba)
acc = accuracy_score(y_test, y_pred)
f1 = f1_score(y_test, y_pred)
precision = precision_score(y_test, y_pred)
recall = recall_score(y_test, y_pred)

# --- Inference latency (single row) ---
sample = X_test.iloc[[0]]
n_latency_runs = 100
t0 = time.time()
for _ in range(n_latency_runs):
    model.predict(sample)
latency_ms = (time.time() - t0) / n_latency_runs * 1000

# --- Inference throughput (1000 rows) ---
batch = X_test.iloc[:1000] if len(X_test) >= 1000 else X_test
t0 = time.time()
model.predict(batch)
elapsed = time.time() - t0
throughput = len(batch) / elapsed

results = {
    "load_data_time_sec": round(load_time, 4),
    "training_time_sec": round(train_time, 4),
    "best_iteration": int(best_iteration),
    "auc_roc": round(float(auc), 6),
    "accuracy": round(float(acc), 6),
    "f1_score": round(float(f1), 6),
    "precision": round(float(precision), 6),
    "recall": round(float(recall), 6),
    "inference_latency_ms_per_row": round(latency_ms, 4),
    "inference_throughput_rows_per_sec": round(throughput, 2),
    "train_rows": int(len(X_train)),
    "test_rows": int(len(X_test)),
}

print(json.dumps(results, indent=2))

with open("benchmark_result.json", "w") as f:
    json.dump(results, f, indent=2)
