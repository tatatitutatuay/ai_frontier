#!/usr/bin/env python3
# -*- coding: utf-8 -*-

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
import tensorflow as tf

from sklearn.model_selection import StratifiedKFold
from sklearn.metrics import (
    confusion_matrix, accuracy_score,
    precision_score, recall_score, f1_score,
    roc_curve
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Dense, Dropout, Flatten
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

# ================= CONFIG =================
SEED = 42
K = 5
EPOCHS = 30
BATCH_SIZE = 16
LR = 1e-4

DIM_X = 128
DIM_Y = 60
INPUT_SHAPE = (DIM_X, DIM_Y, 1)

DATA_DIR = Path("/lustrefs/disk/home/tsuttiwa/ThaiSpoof/final/mfcc/")
OUT_DIR  = Path("/lustrefs/disk/home/tsuttiwa/ThaiSpoof/final/result/mfcc_lcnn_kfold/")
OUT_DIR.mkdir(parents=True, exist_ok=True)

TRAIN_G = DATA_DIR / "MFCC_Train_genuine_3000.pkl"
TRAIN_S = DATA_DIR / "MFCC_Train_spoof_3000.pkl"
TEST_G  = DATA_DIR / "MFCC_Test_genuine_1500.pkl"
TEST_S  = DATA_DIR / "MFCC_Test_spoof_1500.pkl"

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

# ================= PREPROCESS =================
def resize_repeat(mat):
    t, f = mat.shape
    while t < DIM_X:
        mat = np.concatenate([mat, mat], axis=0)
        t, f = mat.shape
    return mat[:DIM_X, :DIM_Y]

def stack_resize(mats):
    out = np.zeros((len(mats), DIM_X, DIM_Y), dtype=np.float32)
    for i, m in enumerate(mats):
        out[i] = resize_repeat(m)
    return out

# ================= MODEL =================
def build_lcnn():
    model = Sequential([
        Conv2D(64, (5,5), activation='relu', input_shape=INPUT_SHAPE),
        MaxPooling2D((2,2)),

        Conv2D(128, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Conv2D(256, (3,3), activation='relu'),
        MaxPooling2D((2,2)),

        Flatten(),
        Dense(512, activation='relu'),
        Dropout(0.5),
        Dense(1, activation='sigmoid')  # binary output
    ])

    model.compile(
        optimizer=Adam(LR),
        loss='binary_crossentropy',
        metrics=['accuracy']
    )
    return model

# ================= METRICS =================
def compute_eer(y_true, scores):
    fpr, tpr, _ = roc_curve(y_true, scores)
    fnr = 1 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    return float((fpr[idx] + fnr[idx]) / 2)

# ================= MAIN =================
def main():
    print("Loading data...")
    train_g = load_pkl(TRAIN_G)
    train_s = load_pkl(TRAIN_S)
    test_g  = load_pkl(TEST_G)
    test_s  = load_pkl(TEST_S)

    Xg = stack_resize(train_g)
    Xs = stack_resize(train_s)

    X = np.concatenate([Xg, Xs], axis=0)[..., np.newaxis]
    y = np.concatenate([
        np.zeros(len(Xg), dtype=np.int32),
        np.ones(len(Xs), dtype=np.int32)
    ])

    Xg_t = stack_resize(test_g)
    Xs_t = stack_resize(test_s)
    X_test = np.concatenate([Xg_t, Xs_t], axis=0)[..., np.newaxis]
    y_test = np.concatenate([
        np.zeros(len(Xg_t), dtype=np.int32),
        np.ones(len(Xs_t), dtype=np.int32)
    ])

    skf = StratifiedKFold(n_splits=K, shuffle=True, random_state=SEED)
    results = []

    for fold, (tr, va) in enumerate(skf.split(X, y), 1):
        print(f"\n===== FOLD {fold}/{K} =====")
        print("train balance:", np.bincount(y[tr]))
        print("val   balance:", np.bincount(y[va]))

        model = build_lcnn()

        callbacks = [
            EarlyStopping(monitor='val_loss', patience=8, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3)
        ]

        model.fit(
            X[tr], y[tr],
            validation_data=(X[va], y[va]),
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            verbose=2,
            callbacks=callbacks
        )
        # ===== TRAIN metrics =====
        p_tr = model.predict(X[tr])
        p_tr_cls = (p_tr >= 0.5).astype(np.int32)
        
        train_acc  = accuracy_score(y[tr], p_tr_cls)
        train_prec = precision_score(y[tr], p_tr_cls)
        train_rec  = recall_score(y[tr], p_tr_cls)
        train_f1   = f1_score(y[tr], p_tr_cls)
        train_eer  = compute_eer(y[tr], p_tr.ravel())

        # ===== validation =====
        pv = model.predict(X[va]).ravel()
        pv_cls = (pv >= 0.5).astype(np.int32)
        eer_v = compute_eer(y[va], pv)

        # ===== test =====
        pt = model.predict(X_test).ravel()
        pt_cls = (pt >= 0.5).astype(np.int32)
        eer_t = compute_eer(y_test, pt)

        cm = confusion_matrix(y_test, pt_cls)
        acc = accuracy_score(y_test, pt_cls)
        prec = precision_score(y_test, pt_cls)
        rec = recall_score(y_test, pt_cls)
        f1 = f1_score(y_test, pt_cls)

        model.save(OUT_DIR / f"lcnn_fold{fold}.h5")

        tn, fp, fn, tp = cm.ravel()
        results.append({
            "fold": fold,
        
            # ===== TRAIN =====
            "train_acc":  train_acc,
            "train_prec": train_prec,
            "train_rec":  train_rec,
            "train_f1":   train_f1,
            "train_eer":  train_eer,
        
            # ===== VALIDATION =====
            "val_eer": eer_v,
        
            # ===== TEST =====
            "test_acc": acc,
            "test_prec": prec,
            "test_rec": rec,
            "test_f1": f1,
            "test_eer": eer_t,
        
            # ===== TEST CONFUSION =====
            "test_tn": tn,
            "test_fp": fp,
            "test_fn": fn,
            "test_tp": tp
        })


    pd.DataFrame(results).to_csv(OUT_DIR / "kfold_metrics.csv", index=False)
    print("Saved kfold_metrics.csv")


if __name__ == '__main__':
    main()
