# -*- coding: utf-8 -*-
import os
import io
import pickle
import numpy as np
import random
import csv
from datetime import datetime
from pathlib import Path
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
import matplotlib.pyplot as plt

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    confusion_matrix,
)

# ----------------- Repro & Threading (set BEFORE TF import) -----------------
SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

# set seeds
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ----------------------------- Config -----------------------------
DIM_X = 128
DIM_Y = 60
INPUT_SHAPE = (DIM_X, DIM_Y, 1)
NUM_CLASSES = 2

# ปรับ PATH ให้ตรงเครื่องคุณ
PKL_ROOT = "/lustrefs/disk/home/tsuttiwa/ThaiSpoof/Data_train_mfcc/"
TRAIN_SETS = ["setA", "setB", "setC", "setD"]
TEST_SET   = "Test"

BASE_DIR     = "/lustrefs/disk/home/tsuttiwa/ThaiSpoof"
EXPERIMENT   = "mfcc_lcnn_2"
RESULTS_DIR  = os.path.join(BASE_DIR, "Results", EXPERIMENT)
MODELS_DIR   = os.path.join(BASE_DIR, "Model",   EXPERIMENT)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

RESULTS_CSV  = os.path.join(RESULTS_DIR, "mfcc_lcnn_4fold_results.csv")
RESULTS_TEX  = os.path.join(RESULTS_DIR, "mfcc_lcnn_4fold_table.tex")
CONF_SUM_CSV = os.path.join(RESULTS_DIR, "confusion_summary_per_fold.csv")
CONF_AGG_CSV = os.path.join(RESULTS_DIR, "confusion_aggregate_sum.csv")

BATCH_SIZE = 16
EPOCHS     = 100

# ----------------------------- Utils: pickle loader ------------------------------
class _NumpyCoreCompatUnpickler(pickle.Unpickler):
    def find_class(self, module, name):
        if module.startswith("numpy._core"):
            module = module.replace("numpy._core", "numpy.core", 1)
        return super().find_class(module, name)

def _pickle_load_compat(fobj) -> object:
    if not isinstance(fobj, (io.BufferedReader, io.BytesIO)):
        fobj = io.BufferedReader(fobj)
    return _NumpyCoreCompatUnpickler(fobj).load()

def load_list_pkl_strict(path: str, min_bytes: int = 1024):
    if not os.path.exists(path):
        raise FileNotFoundError(f"Missing file: {path}")
    size = os.path.getsize(path)
    if size < min_bytes:
        raise ValueError(f"File too small ({size} bytes): {path}")
    with open(path, "rb") as f:
        obj = _pickle_load_compat(f)
    if isinstance(obj, np.ndarray):
        obj = obj.tolist()
    if not isinstance(obj, (list, tuple)) or len(obj) == 0:
        raise ValueError(f"Unexpected/empty dataset in {path}")
    clean = []
    for i, x in enumerate(obj):
        x = np.asarray(x, dtype=np.float32)
        if x.ndim != 2:
            raise ValueError(f"Sample {i} must be 2D (T,F), got shape {x.shape}")
        clean.append(x)
    return clean

def pad_truncate_to_rect(mat, dim_x, dim_y):
    x = np.array(mat, dtype=np.float32)
    t, f = x.shape

    # แกนเวลา
    if t < dim_x:
        orig = x.copy()
        while x.shape[0] < dim_x:
            need = dim_x - x.shape[0]
            x = np.concatenate(
                [x, orig if need >= orig.shape[0] else orig[:need]], axis=0
            )
    elif t > dim_x:
        x = x[:dim_x, :]

    # แกนความถี่
    if f < dim_y:
        pad = np.zeros((x.shape[0], dim_y - f), dtype=np.float32)
        x = np.concatenate([x, pad], axis=1)
    elif f > dim_y:
        x = x[:, :dim_y]
    return x

def batch_rectify(list_of_mats, dim_x, dim_y):
    out = np.zeros((len(list_of_mats), dim_x, dim_y), dtype=np.float32)
    for i, m in enumerate(list_of_mats):
        out[i] = pad_truncate_to_rect(m, dim_x, dim_y)
    return out

def to_channel_last(x):
    return x[..., None].astype(np.float32, copy=False)

def make_labels(n_genuine, n_spoof):
    # 0 = genuine, 1 = spoof
    return np.concatenate(
        [np.zeros(n_genuine, dtype=np.int32),
         np.ones(n_spoof,   dtype=np.int32)]
    )

def sanity_checks(x, y, name=""):
    print(f"[{name}] x:", x.shape, x.dtype,
          "min/max", float(np.nanmin(x)), float(np.nanmax(x)))
    print(f"[{name}] y:", y.shape, y.dtype, "unique", np.unique(y))
    assert x.ndim == 4 and x.shape[1:] == (DIM_X, DIM_Y, 1)
    assert y.ndim == 1 and len(x) == len(y) and len(x) > 0
    assert set(np.unique(y)).issubset({0, 1})
    assert np.isfinite(x).all() and np.isfinite(y).all()

# ----------------------------- LightCNN (MFM) implementation -----------------------------
def mfm_conv(x, out_channels, kernel_size=(3,3), strides=(1,1), padding="same", name=None):
    conv = layers.Conv2D(
        filters=out_channels * 2,
        kernel_size=kernel_size,
        strides=strides,
        padding=padding,
        kernel_initializer=keras.initializers.HeNormal(seed=SEED),
        name=(None if name is None else name + "_conv")
    )(x)
    bn = layers.BatchNormalization(name=(None if name is None else name + "_bn"))(conv)

    def mfm_fn(t):
        c = tf.shape(t)[-1] // 2
        a = t[..., :c]
        b = t[..., c:]
        return tf.maximum(a, b)

    m = layers.Lambda(mfm_fn, name=(None if name is None else name + "_mfm"))(bn)
    return m

def build_lcnn_model(input_shape=(128,60,1), num_classes=2):
    he_init = keras.initializers.HeNormal(seed=SEED)
    inputs = layers.Input(shape=input_shape)

    # Block1: larger kernel like AlexNet conv1
    x = mfm_conv(inputs, out_channels=48, kernel_size=(5,5), padding="same", name="conv1")
    x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool1")(x)

    # Block2
    x = mfm_conv(x, out_channels=96, kernel_size=(3,3), padding="same", name="conv2")
    x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool2")(x)

    # Block3: stacked convs
    x = mfm_conv(x, out_channels=192, kernel_size=(3,3), padding="same", name="conv3a")
    x = mfm_conv(x, out_channels=192, kernel_size=(3,3), padding="same", name="conv3b")

    # Block4
    x = mfm_conv(x, out_channels=128, kernel_size=(3,3), padding="same", name="conv4")
    x = layers.MaxPooling2D(pool_size=(2,2), strides=(2,2), name="pool4")(x)

    # Head
    x = layers.GlobalAveragePooling2D(name="gap")(x)
    x = layers.Dense(256, kernel_initializer=he_init, name="fc1")(x)
    x = layers.BatchNormalization(name="fc1_bn")(x)
    x = layers.Activation("relu", name="fc1_act")(x)
    x = layers.Dropout(0.5, name="drop_fc1")(x)

    outputs = layers.Dense(num_classes, activation="softmax", name="pred")(x)

    model = keras.Model(inputs=inputs, outputs=outputs, name="MFCC_LightCNN")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=1e-4),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ----------------------------- Metrics helper -----------------------------
def compute_eer(y_true, y_score):
    fpr, tpr, thresholds = roc_curve(y_true, y_score, pos_label=1)
    fnr = 1.0 - tpr
    idx = np.nanargmin(np.abs(fnr - fpr))
    eer = (fpr[idx] + fnr[idx]) / 2.0
    thr = thresholds[idx]
    return float(eer), float(thr)

def compute_all_metrics(y_true, y_pred, y_score_pos):
    acc  = accuracy_score(y_true, y_pred)
    bacc = balanced_accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred, zero_division=0)
    rec  = recall_score(y_true, y_pred, zero_division=0)
    f1   = f1_score(y_true, y_pred, zero_division=0)
    eer, thr = compute_eer(y_true, y_score_pos)
    return {
        "accuracy":      float(acc),
        "balanced_acc":  float(bacc),
        "precision":     float(prec),
        "recall":        float(rec),
        "f1":            float(f1),
        "eer":           float(eer),
        "eer_thr":       float(thr),
    }

def fmt(x):
    if x is None:
        return "-"
    return f"{x:.4f}"

# ----------------------------- Helpers: load per set ------------------------------
def _find_single_pkl(pkl_dir: Path, pattern: str) -> Path:
    matches = sorted(pkl_dir.glob(pattern))
    if not matches:
        raise FileNotFoundError(f"No PKL matched in {pkl_dir} with pattern {pattern}")
    if len(matches) > 1:
        return matches[-1]
    return matches[0]

def load_set_xy(set_name: str):
    set_dir = Path(PKL_ROOT) / set_name
    if set_name.lower() == "test":
        split_label = "Test"
    else:
        split_label = "Train"

    pkl_g = _find_single_pkl(set_dir, f"MFCC_{set_name}_genuine_{split_label}_*.pkl")
    pkl_s = _find_single_pkl(set_dir, f"MFCC_{set_name}_spoof_{split_label}_*.pkl")

    print(f"[{set_name}] genuine PKL:", pkl_g)
    print(f"[{set_name}] spoof   PKL:", pkl_s)

    list_g = load_list_pkl_strict(str(pkl_g))
    list_s = load_list_pkl_strict(str(pkl_s))

    Xg = batch_rectify(list_g, DIM_X, DIM_Y)
    Xs = batch_rectify(list_s, DIM_X, DIM_Y)

    return Xg, Xs

# ----------------------------- IO helpers for confusion & plotting -----------------------------
def save_confusion_matrix(cm, path):
    np.savetxt(path, cm, fmt='%d', delimiter=',')
    print(f"Saved confusion matrix to: {path}")

def save_confusion_matrix_norm(cm, path):
    with np.errstate(all='ignore'):
        row_sums = cm.sum(axis=1, keepdims=True)
        cm_norm = cm.astype(np.float64) / np.where(row_sums == 0, 1, row_sums)
    np.savetxt(path, cm_norm, fmt='%.6f', delimiter=',')
    print(f"Saved normalized confusion matrix to: {path}")

def plot_and_save_cm(cm, path_png, title="Confusion matrix"):
    # cm: 2x2 matrix (counts)
    cm_norm = cm.astype(np.float64)
    row_sums = cm_norm.sum(axis=1, keepdims=True)
    cm_norm = cm_norm / np.where(row_sums == 0, 1, row_sums)
    fig, ax = plt.subplots(figsize=(4,4))
    im = ax.imshow(cm_norm, interpolation='nearest')
    ax.set_title(title)
    ax.set_xlabel('Predicted')
    ax.set_ylabel('True')
    ax.set_xticks([0,1]); ax.set_yticks([0,1])
    ax.set_xticklabels(['genuine','spoof'])
    ax.set_yticklabels(['genuine','spoof'])
    # show counts and normalized in cells
    for i in range(2):
        for j in range(2):
            count = int(cm[i,j])
            frac = cm_norm[i,j]
            ax.text(j, i, f"{count}\n{frac:.2f}", ha="center", va="center")
    fig.tight_layout()
    fig.savefig(path_png)
    plt.close(fig)
    print(f"Saved confusion heatmap to: {path_png}")

# ----------------------------- Main -----------------------------
def main():
    print("=== Loading per-set MFCC PKL files ===")

    # โหลดทุก setA–D + Test
    X_genuine_sets = {}
    X_spoof_sets   = {}
    for s in TRAIN_SETS + [TEST_SET]:
        Xg, Xs = load_set_xy(s)
        X_genuine_sets[s] = Xg
        X_spoof_sets[s]   = Xs

    # Test set
    X_test_2d = np.concatenate(
        [X_genuine_sets[TEST_SET], X_spoof_sets[TEST_SET]],
        axis=0
    )
    y_test = make_labels(
        len(X_genuine_sets[TEST_SET]),
        len(X_spoof_sets[TEST_SET]),
    )
    X_test = to_channel_last(X_test_2d)
    sanity_checks(X_test, y_test, "Test")

    # แผน fold:
    fold_plan = []
    for i in range(len(TRAIN_SETS)):
        val_set = TRAIN_SETS[i]
        train_sets = [s for s in TRAIN_SETS if s != val_set]
        fold_plan.append((val_set, train_sets))

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau

    early_stopper = EarlyStopping(
        monitor="val_loss",
        patience=10,
        verbose=1,
        restore_best_weights=True,
    )

    reduce_lr = ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        verbose=1,
        min_lr=1e-6,
    )

    fold_results = []
    test_probas_from_folds = []

    # prepare CSV summary header for per-fold confusion/metrics
    with open(CONF_SUM_CSV, "w", newline="", encoding="utf-8") as fsum:
        writer = csv.writer(fsum)
        writer.writerow([
            "fold", "split",
            "TN","FP","FN","TP",
            "accuracy","balanced_acc","precision","recall","f1","eer"
        ])

    for fold_idx, (val_set, train_sets) in enumerate(fold_plan, start=1):
        print(f"\n===== Fold {fold_idx} / 4 =====")
        print(f"  Train sets: {train_sets}")
        print(f"  Val   set : {val_set}")

        # ----- สร้าง Train -----
        X_tr_list = []
        y_tr_list = []
        for s in train_sets:
            Xg = X_genuine_sets[s]
            Xs = X_spoof_sets[s]
            X_tr_list.append(Xg)
            X_tr_list.append(Xs)
            y_tr_list.append(make_labels(len(Xg), len(Xs)))

        X_tr_2d = np.concatenate(X_tr_list, axis=0)
        y_tr    = np.concatenate(y_tr_list, axis=0)

        # ----- Validation -----
        Xg_val = X_genuine_sets[val_set]
        Xs_val = X_spoof_sets[val_set]
        X_val_2d = np.concatenate([Xg_val, Xs_val], axis=0)
        y_val    = make_labels(len(Xg_val), len(Xs_val))

        X_tr  = to_channel_last(X_tr_2d)
        X_val = to_channel_last(X_val_2d)

        if fold_idx == 1:
            sanity_checks(X_tr,  y_tr,  "Fold1-Train")
            sanity_checks(X_val, y_val, "Fold1-Val")

        # ----- Model (LightCNN with MFM) -----
        model = build_lcnn_model(INPUT_SHAPE, NUM_CLASSES)
        model_path = os.path.join(MODELS_DIR, f"mfcc_lcnn_fold{fold_idx}.h5")

        history = model.fit(
            X_tr, y_tr,
            epochs=EPOCHS,
            batch_size=BATCH_SIZE,
            shuffle=True,
            validation_data=(X_val, y_val),
            callbacks=[early_stopper, reduce_lr],
            verbose=1,
        )

        # Loss & acc จาก Keras evaluate
        train_loss, train_acc = model.evaluate(X_tr, y_tr, verbose=0)
        val_loss,   val_acc   = model.evaluate(X_val, y_val, verbose=0)

        # ใช้ proba เพื่อคำนวณ metrics อื่น + EER
        prob_tr  = model.predict(X_tr,  batch_size=64, verbose=0)
        prob_val = model.predict(X_val, batch_size=64, verbose=0)

        y_pred_tr  = np.argmax(prob_tr,  axis=1)
        y_pred_val = np.argmax(prob_val, axis=1)

        # ------------------- CONFUSION MATRICES -------------------
        cm_tr = confusion_matrix(y_tr, y_pred_tr, labels=[0,1])
        cm_val = confusion_matrix(y_val, y_pred_val, labels=[0,1])

        # Save matrices (counts + normalized)
        save_confusion_matrix(cm_tr, os.path.join(RESULTS_DIR, f"confusion_fold{fold_idx}_train.csv"))
        save_confusion_matrix_norm(cm_tr, os.path.join(RESULTS_DIR, f"confusion_fold{fold_idx}_train_norm.csv"))
        save_confusion_matrix(cm_val, os.path.join(RESULTS_DIR, f"confusion_fold{fold_idx}_val.csv"))
        save_confusion_matrix_norm(cm_val, os.path.join(RESULTS_DIR, f"confusion_fold{fold_idx}_val_norm.csv"))

        # Save heatmaps
        plot_and_save_cm(cm_tr, os.path.join(RESULTS_DIR, f"confusion_fold{fold_idx}_train.png"), title=f"Fold{fold_idx} Train CM")
        plot_and_save_cm(cm_val, os.path.join(RESULTS_DIR, f"confusion_fold{fold_idx}_val.png"),   title=f"Fold{fold_idx} Val CM")

        # extract TN,FP,FN,TP
        try:
            TN_tr, FP_tr, FN_tr, TP_tr = cm_tr.ravel()
        except Exception:
            # handle degenerate shapes (e.g., missing class)
            flat = cm_tr.flatten()
            if flat.size == 4:
                TN_tr, FP_tr, FN_tr, TP_tr = flat
            else:
                # fallback: fill zeros
                TN_tr = FP_tr = FN_tr = TP_tr = 0

        try:
            TN_val, FP_val, FN_val, TP_val = cm_val.ravel()
        except Exception:
            flat = cm_val.flatten()
            if flat.size == 4:
                TN_val, FP_val, FN_val, TP_val = flat
            else:
                TN_val = FP_val = FN_val = TP_val = 0

        # compute metrics (from predictions)
        metrics_tr  = compute_all_metrics(y_tr,  y_pred_tr,  prob_tr[:, 1])
        metrics_val = compute_all_metrics(y_val, y_pred_val, prob_val[:, 1])

        # append per-fold results for table (same format as before)
        fold_results.append({
            "fold": fold_idx,
            "train": {
                "loss": train_loss,
                "accuracy": metrics_tr["accuracy"],
                "balanced_acc": metrics_tr["balanced_acc"],
                "precision": metrics_tr["precision"],
                "recall": metrics_tr["recall"],
                "f1": metrics_tr["f1"],
                "eer": metrics_tr["eer"],
                "TN": int(TN_tr), "FP": int(FP_tr), "FN": int(FN_tr), "TP": int(TP_tr),
            },
            "val": {
                "loss": val_loss,
                "accuracy": metrics_val["accuracy"],
                "balanced_acc": metrics_val["balanced_acc"],
                "precision": metrics_val["precision"],
                "recall": metrics_val["recall"],
                "f1": metrics_val["f1"],
                "eer": metrics_val["eer"],
                "TN": int(TN_val), "FP": int(FP_val), "FN": int(FN_val), "TP": int(TP_val),
            },
        })

        # write per-fold confusion/metrics row to summary CSV
        with open(CONF_SUM_CSV, "a", newline="", encoding="utf-8") as fsum:
            writer = csv.writer(fsum)
            writer.writerow([
                fold_idx, "train",
                int(TN_tr), int(FP_tr), int(FN_tr), int(TP_tr),
                fmt(metrics_tr["accuracy"]), fmt(metrics_tr["balanced_acc"]),
                fmt(metrics_tr["precision"]), fmt(metrics_tr["recall"]),
                fmt(metrics_tr["f1"]), fmt(metrics_tr["eer"]),
            ])
            writer.writerow([
                fold_idx, "val",
                int(TN_val), int(FP_val), int(FN_val), int(TP_val),
                fmt(metrics_val["accuracy"]), fmt(metrics_val["balanced_acc"]),
                fmt(metrics_val["precision"]), fmt(metrics_val["recall"]),
                fmt(metrics_val["f1"]), fmt(metrics_val["eer"]),
            ])

        # เก็บ proba ของ Test จากแต่ละ fold ไว้ทำ average ensemble
        prob_test = model.predict(X_test, batch_size=64, verbose=0)
        test_probas_from_folds.append(prob_test)

        # เซฟโมเดลของแต่ละ fold
        model.save(model_path)
        print(f"Saved model for fold {fold_idx} to: {model_path}")

        # เคลียร์ session ช่วยลด memory
        keras.backend.clear_session()
        tf.compat.v1.reset_default_graph()

    # ---------------- Test metrics (ensemble จากทุก fold) ----------------
    print("\n=== Evaluate on Test set (ensemble of folds) ===")
    avg_test_proba = np.mean(test_probas_from_folds, axis=0)
    y_pred_test    = np.argmax(avg_test_proba, axis=1)
    metrics_test   = compute_all_metrics(y_test, y_pred_test, avg_test_proba[:, 1])

    # ---------------- Print summary table ----------------
    print("\n===== 4-fold summary (for LaTeX/CSV table) =====")
    header = [
        "4-fold cross-validation",
        "Split",
        "Loss", "Accuracy", "Balanced Acc",
        "Precision", "Recall", "F1", "EER",
    ]
    print("\t".join(header))

    rows_for_csv = []

    for r in fold_results:
        k = r["fold"]

        # Train
        row_train = [
            f"Round{k}", "Train",
            fmt(r["train"]["loss"]),
            fmt(r["train"]["accuracy"]),
            fmt(r["train"]["balanced_acc"]),
            fmt(r["train"]["precision"]),
            fmt(r["train"]["recall"]),
            fmt(r["train"]["f1"]),
            fmt(r["train"]["eer"]),
        ]
        print("\t".join(row_train))
        rows_for_csv.append(row_train)

        # Validation
        row_val = [
            "", "Validation",
            fmt(r["val"]["loss"]),
            fmt(r["val"]["accuracy"]),
            fmt(r["val"]["balanced_acc"]),
            fmt(r["val"]["precision"]),
            fmt(r["val"]["recall"]),
            fmt(r["val"]["f1"]),
            fmt(r["val"]["eer"]),
        ]
        print("\t".join(row_val))
        rows_for_csv.append(row_val)

    # Test row (ไม่ใส่ loss)
    row_test = [
        "", "Test",
        "-",
        fmt(metrics_test["accuracy"]),
        fmt(metrics_test["balanced_acc"]),
        fmt(metrics_test["precision"]),
        fmt(metrics_test["recall"]),
        fmt(metrics_test["f1"]),
        fmt(metrics_test["eer"]),
    ]
    print("\t".join(row_test))
    rows_for_csv.append(row_test)

    # ---------------- Save CSV (summary table) ----------------
    with open(RESULTS_CSV, "w", newline="", encoding="utf-8") as f:
        writer = csv.writer(f)
        writer.writerow(header)
        writer.writerows(rows_for_csv)
    print(f"\nSaved CSV to: {RESULTS_CSV}")

    # ---------------- Save LaTeX table ----------------
    with open(RESULTS_TEX, "w", encoding="utf-8") as f:
        f.write("% Auto-generated on " + datetime.now().strftime("%Y-%m-%d %H:%M:%S") + "\n")
        f.write("\\begin{table}[t]\n")
        f.write("\\centering\n")
        f.write("\\small\n")
        f.write("\\begin{tabular}{llccccccc}\n")
        f.write("\\hline\n")
        f.write("4-fold cross-validation & Split & Loss & Accuracy & Balanced Acc & Precision & Recall & F1 & EER\\\\\n")
        f.write("\\hline\n")
        for row in rows_for_csv:
            f.write(
                f"{row[0]} & {row[1]} & {row[2]} & {row[3]} & "
                f"{row[4]} & {row[5]} & {row[6]} & {row[7]} & {row[8]}\\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{LFCC/ MFCC LightCNN 4-fold cross-validation using setA--setD as folds and Test as evaluation set.}\n")
        f.write("\\label{tab:lfcc_lcnn_4fold}\n")
        f.write("\\end{table}\n")
    print(f"Saved LaTeX table to: {RESULTS_TEX}")

    # ---------------- Aggregate confusion matrices across folds (sum) ----------------
    agg_train = np.zeros((2,2), dtype=np.int64)
    agg_val   = np.zeros((2,2), dtype=np.int64)
    for r in fold_results:
        agg_train += np.array([[r["train"]["TN"], r["train"]["FP"]],[r["train"]["FN"], r["train"]["TP"]]])
        agg_val   += np.array([[r["val"]["TN"],   r["val"]["FP"]],[r["val"]["FN"],   r["val"]["TP"]]])

    # save aggregated
    save_confusion_matrix(agg_train, CONF_AGG_CSV.replace(".csv", "_train_sum.csv"))
    save_confusion_matrix_norm(agg_train, CONF_AGG_CSV.replace(".csv", "_train_sum_norm.csv"))
    save_confusion_matrix(agg_val,   CONF_AGG_CSV.replace(".csv", "_val_sum.csv"))
    save_confusion_matrix_norm(agg_val,   CONF_AGG_CSV.replace(".csv", "_val_sum_norm.csv"))

    plot_and_save_cm(agg_train, os.path.join(RESULTS_DIR, "confusion_train_aggregate_sum.png"), title="Aggregate Train CM (sum over folds)")
    plot_and_save_cm(agg_val,   os.path.join(RESULTS_DIR, "confusion_val_aggregate_sum.png"),   title="Aggregate Val CM (sum over folds)")

    # write aggregate counts to CSV (simple)
    with open(CONF_AGG_CSV, "w", newline="", encoding="utf-8") as fa:
        writer = csv.writer(fa)
        writer.writerow(["split","TN","FP","FN","TP"])
        writer.writerow(["train", int(agg_train[0,0]), int(agg_train[0,1]), int(agg_train[1,0]), int(agg_train[1,1])])
        writer.writerow(["val",   int(agg_val[0,0]),   int(agg_val[0,1]),   int(agg_val[1,0]),   int(agg_val[1,1])])
    print(f"Saved aggregate confusion CSV to: {CONF_AGG_CSV}")

if __name__ == "__main__":
    main()
