#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
from pathlib import Path
import numpy as np
import pandas as pd
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve
)

from classification_models.keras import Classifiers

# ================= CONFIG =================
DATA_DIR = Path("/lustrefs/disk/home/tsuttiwa/ThaiSpoof/final/mfcc/")
OUT_DIR  = Path("/lustrefs/disk/home/tsuttiwa/ThaiSpoof/final/result/mfcc_resnet34_kfold")
OUT_DIR.mkdir(parents=True, exist_ok=True)

DIM_X = 723
DIM_Y = 60
INPUT_SHAPE = (DIM_Y, DIM_X, 1)

K = 5
EPOCHS = 30
BATCH = 16
LR = 1e-4
SEED = 42

np.random.seed(SEED)
tf.random.set_seed(SEED)

# ================= IO =================
class _NumpyCoreCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

def load_pkl(path):
    with open(path, "rb") as f:
        obj = _NumpyCoreCompatUnpickler(f).load()
    return [np.asarray(x, np.float32) for x in obj]

# ================= FEATURE =================
def repeat_to_dim_x(mat, dim_x):
    t, f = mat.shape
    while t < dim_x:
        mat = np.concatenate([mat, mat], axis=0)
        t = mat.shape[0]
    return mat[:dim_x, :DIM_Y]

def prepare(feats):
    out = np.zeros((len(feats), DIM_X, DIM_Y), np.float32)
    for i, f in enumerate(feats):
        out[i] = repeat_to_dim_x(f, DIM_X)
    out = np.transpose(out, (0, 2, 1))
    return out[..., np.newaxis]

# ================= MODEL =================
def build_resnet34():
    ResNet34, _ = Classifiers.get("resnet34")
    model = ResNet34(
        input_shape=INPUT_SHAPE,
        classes=2,
        weights=None
    )
    model.compile(
        optimizer=keras.optimizers.Adam(LR),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"]
    )
    return model

# ================= METRICS =================
def compute_eer(y_true, prob):
    score = prob[:, 1]
    fpr, tpr, _ = roc_curve(y_true, score)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)

def calc_metrics(y_true, prob):
    pred = prob.argmax(axis=1)
    cm = confusion_matrix(y_true, pred)
    tn, fp, fn, tp = cm.ravel()

    return {
        "acc": accuracy_score(y_true, pred),
        "prec": precision_score(y_true, pred),
        "rec": recall_score(y_true, pred),
        "f1": f1_score(y_true, pred),
        "eer": compute_eer(y_true, prob),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }

# ================= MAIN =================
def main():
    print("Loading data...")
    train_g = load_pkl(DATA_DIR / "MFCC_Train_genuine_3000.pkl")
    train_s = load_pkl(DATA_DIR / "MFCC_Train_spoof_3000.pkl")
    test_g  = load_pkl(DATA_DIR / "MFCC_Test_genuine_1500.pkl")
    test_s  = load_pkl(DATA_DIR / "MFCC_Test_spoof_1500.pkl")

    Xg = prepare(train_g)
    Xs = prepare(train_s)
    X  = np.concatenate([Xg, Xs], axis=0)
    y  = np.concatenate([
        np.zeros(len(Xg), np.int32),
        np.ones(len(Xs), np.int32)
    ])

    Xg_t = prepare(test_g)
    Xs_t = prepare(test_s)
    X_test = np.concatenate([Xg_t, Xs_t], axis=0)
    y_test = np.concatenate([
        np.zeros(len(Xg_t), np.int32),
        np.ones(len(Xs_t), np.int32)
    ])

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    results = []

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"\n===== FOLD {fold}/{K} =====")

        model = build_resnet34()

        callbacks = [
            EarlyStopping(monitor="val_loss", patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor="val_loss", factor=0.5, patience=3)
        ]

        model.fit(
            X[tr], y[tr],
            validation_data=(X[va], y[va]),
            epochs=EPOCHS,
            batch_size=BATCH,
            verbose=2,
            callbacks=callbacks
        )

        # ===== TRAIN METRICS =====
        prob_tr = model.predict(X[tr])
        train_m = calc_metrics(y[tr], prob_tr)

        # ===== TEST METRICS =====
        prob_te = model.predict(X_test)
        test_m = calc_metrics(y_test, prob_te)

        print("TRAIN:", train_m)
        print("TEST :", test_m)

        results.append({
            "fold": fold,

            **{f"train_{k}": v for k, v in train_m.items()},
            **{f"test_{k}": v for k, v in test_m.items()},
        })

        model.save(OUT_DIR / f"resnet34_fold{fold}.h5")

    pd.DataFrame(results).to_csv(OUT_DIR / "kfold_summary.csv", index=False)
    print("Saved summary CSV")

if __name__ == "__main__":
    main()
