# -*- coding: utf-8 -*-

import os
import io
import pickle
import numpy as np
import random
import csv
from datetime import datetime
from pathlib import Path

# ----------------- Repro & Threading (set BEFORE TF import) -----------------
SEED = 42
os.environ.setdefault("PYTHONHASHSEED", str(SEED))
os.environ.setdefault("TF_CPP_MIN_LOG_LEVEL", "2")
os.environ.setdefault("OMP_NUM_THREADS", "1")
os.environ.setdefault("OPENBLAS_NUM_THREADS", "1")
os.environ.setdefault("MKL_NUM_THREADS", "1")
os.environ.setdefault("NUMEXPR_NUM_THREADS", "1")
os.environ.setdefault("TF_DETERMINISTIC_OPS", "1")

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_curve,
    confusion_matrix,
)
import matplotlib.pyplot as plt

# ตั้งค่า seed ให้ครบ
random.seed(SEED)
np.random.seed(SEED)
tf.keras.utils.set_random_seed(SEED)
tf.config.threading.set_intra_op_parallelism_threads(1)
tf.config.threading.set_inter_op_parallelism_threads(1)

# ----------------------------- Config -----------------------------
DIM_X = 723
DIM_Y = 60
INPUT_SHAPE = (DIM_X, DIM_Y, 1)
NUM_CLASSES = 2

# รากของฟีเจอร์ที่ extractor สร้าง (โฟลเดอร์ย่อย: setA, setB, setC, setD, Test)
PKL_ROOT = "/lustrefs/disk/home/tsuttiwa/ThaiSpoof/data_fusion train/"
TRAIN_SETS = ["setA", "setB", "setC", "setD"]
TEST_SET   = "Test"

BASE_DIR     = "/lustrefs/disk/home/tsuttiwa/ThaiSpoof"
EXPERIMENT   = "fusion_resnet"
RESULTS_DIR  = os.path.join(BASE_DIR, "Results", EXPERIMENT)
MODELS_DIR   = os.path.join(BASE_DIR, "Model",   EXPERIMENT)
os.makedirs(RESULTS_DIR, exist_ok=True)
os.makedirs(MODELS_DIR,  exist_ok=True)

RESULTS_CSV  = os.path.join(RESULTS_DIR, "lfcc.csv")
RESULTS_TEX  = os.path.join(RESULTS_DIR, "lfcc.tex")

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

# ----------------------------- ResNet34 -----------------------------
def _basic_block(x, filters, s=1, seed=SEED):
    he_init = keras.initializers.HeNormal(seed=seed)
    shortcut = x
    in_channels = x.shape[-1]

    y = layers.Conv2D(filters, 3, strides=s, padding="same", use_bias=False,
                      kernel_initializer=he_init)(x)
    y = layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(y)
    y = layers.Activation("relu")(y)

    y = layers.Conv2D(filters, 3, strides=1, padding="same", use_bias=False,
                      kernel_initializer=he_init)(y)
    y = layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(y)

    if (s != 1) or (in_channels != filters):
        shortcut = layers.Conv2D(filters, 1, strides=s, padding="same",
                                 use_bias=False, kernel_initializer=he_init)(shortcut)
        shortcut = layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(shortcut)

    out = layers.Activation("relu")(layers.Add()([y, shortcut]))
    return out


def _make_layer(x, filters, num_blocks, first_stride, seed=SEED):
    x = _basic_block(x, filters, s=first_stride, seed=seed)
    for _ in range(num_blocks - 1):
        x = _basic_block(x, filters, s=1, seed=seed)
    return x


def build_resnet34_model(input_shape=(1200, 60, 1), num_classes=2):
    he_init = keras.initializers.HeNormal(seed=SEED)
    xavier = keras.initializers.GlorotUniform(seed=SEED)

    inputs = layers.Input(shape=input_shape)
    x = layers.Conv2D(64, 7, strides=2, padding="same", use_bias=False,
                      kernel_initializer=he_init)(inputs)
    x = layers.BatchNormalization(epsilon=1e-5, momentum=0.9)(x)
    x = layers.Activation("relu")(x)
    x = layers.MaxPooling2D(3, strides=2, padding="same")(x)

    x = _make_layer(x,  64, 3, first_stride=1, seed=SEED)
    x = _make_layer(x, 128, 4, first_stride=2, seed=SEED)
    x = _make_layer(x, 256, 6, first_stride=2, seed=SEED)
    x = _make_layer(x, 512, 3, first_stride=2, seed=SEED)

    x = layers.GlobalAveragePooling2D()(x)
    x = layers.Dropout(0.5)(x)

    outputs = layers.Dense(
        num_classes,
        activation="softmax",
        kernel_initializer=xavier,
    )(x)

    model = keras.Model(inputs, outputs, name="lfcc_ResNet34_fusion")
    model.compile(
        optimizer=keras.optimizers.Adam(learning_rate=0.000177828),
        loss="sparse_categorical_crossentropy",
        metrics=["accuracy"],
    )
    return model

# ----------------------------- Metrics helper -----------------------------
def compute_eer(y_true, y_score):
    # y_score = probability of positive class (spoof)
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
        # ถ้ามีหลายไฟล์ เอาตัวท้ายสุด (เช่นสุ่มใหม่ล่าสุด)
        return matches[-1]
    return matches[0]


def load_set_xy(set_name: str):
    set_dir = Path(PKL_ROOT) / set_name
    if set_name.lower() == "test":
        split_label = "Test"
    else:
        split_label = "Train"

    pkl_g = _find_single_pkl(set_dir, f"Fusion_{set_name}_genuine_{split_label}_*.pkl")
    pkl_s = _find_single_pkl(set_dir, f"Fusion_{set_name}_spoof_{split_label}_*.pkl")

    print(f"[{set_name}] genuine PKL:", pkl_g)
    print(f"[{set_name}] spoof   PKL:", pkl_s)

    list_g = load_list_pkl_strict(str(pkl_g))
    list_s = load_list_pkl_strict(str(pkl_s))

    Xg = batch_rectify(list_g, DIM_X, DIM_Y)
    Xs = batch_rectify(list_s, DIM_X, DIM_Y)

    return Xg, Xs

# ----------------------------- Confusion matrix helpers -----------------------------
LABELS = [0, 1]
LABEL_NAMES = ["genuine", "spoof"]


def save_confusion_matrix(cm, path_csv, path_png=None, labels=None, title=None):
    # save csv
    np.savetxt(path_csv, cm, fmt='%d', delimiter=',')

    if path_png is not None:
        plt.figure()
        plt.imshow(cm, interpolation='nearest')
        plt.title(title if title is not None else '')
        plt.xlabel('Predicted')
        plt.ylabel('True')
        ticks = range(len(labels)) if labels is not None else range(cm.shape[0])
        plt.xticks(ticks, labels, rotation=45)
        plt.yticks(ticks, labels)

        # annotate cells
        thresh = cm.max() / 2.0 if cm.size else 0
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                plt.text(j, i, int(cm[i, j]),
                         ha="center", va="center",
                         fontsize=10,
                         color="white" if cm[i, j] > thresh else "black")
        plt.tight_layout()
        plt.savefig(path_png)
        plt.close()

# ----------------------------- Main -----------------------------
def main():
    print("=== Loading per-set LFCC PKL files ===")

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
    # Fold1: val=setA, train=setB+C+D
    # Fold2: val=setB, train=setA+C+D
    # Fold3: val=setC, train=setA+B+D
    # Fold4: val=setD, train=setA+B+C
    fold_plan = []
    for i in range(len(TRAIN_SETS)):
        val_set = TRAIN_SETS[i]
        train_sets = [s for s in TRAIN_SETS if s != val_set]
        fold_plan.append((val_set, train_sets))

    from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau, ModelCheckpoint

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
    confusion_records = []
    test_probas_from_folds = []

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

        # ----- Model -----
        model = build_resnet34_model(INPUT_SHAPE, NUM_CLASSES)
        model_path = os.path.join(MODELS_DIR, f"fusion_resnet34_fold{fold_idx}.h5")

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
        # ---------------- Compute confusion matrices and save ----------------
        cm_tr = confusion_matrix(y_tr, y_pred_tr, labels=LABELS)
        cm_val = confusion_matrix(y_val, y_pred_val, labels=LABELS)

        cm_tr_csv = os.path.join(RESULTS_DIR, f'confusion_fold{fold_idx}_train.csv')
        cm_val_csv = os.path.join(RESULTS_DIR, f'confusion_fold{fold_idx}_val.csv')
        cm_tr_png = os.path.join(RESULTS_DIR, f'confusion_fold{fold_idx}_train.png')
        cm_val_png = os.path.join(RESULTS_DIR, f'confusion_fold{fold_idx}_val.png')

        save_confusion_matrix(cm_tr, cm_tr_csv, cm_tr_png, labels=LABEL_NAMES,
                              title=f'Fold {fold_idx} - Train')
        save_confusion_matrix(cm_val, cm_val_csv, cm_val_png, labels=LABEL_NAMES,
                              title=f'Fold {fold_idx} - Validation')

        print(f"Saved confusion matrices for fold {fold_idx} to: {cm_tr_csv}, {cm_tr_png}, {cm_val_csv}, {cm_val_png}")

        # Extract TN, FP, FN, TP (confusion_matrix returns array [[TN, FP],[FN, TP]])
        tn_tr, fp_tr = int(cm_tr[0,0]), int(cm_tr[0,1])
        fn_tr, tp_tr = int(cm_tr[1,0]), int(cm_tr[1,1])
        tn_val, fp_val = int(cm_val[0,0]), int(cm_val[0,1])
        fn_val, tp_val = int(cm_val[1,0]), int(cm_val[1,1])

        # Append numeric records for later saving
        confusion_records.append({
            'fold': fold_idx,
            'split': 'train',
            'tn': tn_tr, 'fp': fp_tr, 'fn': fn_tr, 'tp': tp_tr,
        })
        confusion_records.append({
            'fold': fold_idx,
            'split': 'val',
            'tn': tn_val, 'fp': fp_val, 'fn': fn_val, 'tp': tp_val,
        })

        metrics_tr  = compute_all_metrics(y_tr,  y_pred_tr,  prob_tr[:, 1])
        metrics_val = compute_all_metrics(y_val, y_pred_val, prob_val[:, 1])


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
            },
            "val": {
                "loss": val_loss,
                "accuracy": metrics_val["accuracy"],
                "balanced_acc": metrics_val["balanced_acc"],
                "precision": metrics_val["precision"],
                "recall": metrics_val["recall"],
                "f1": metrics_val["f1"],
                "eer": metrics_val["eer"],
            },
        })

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

    # Save confusion for test ensemble
    cm_test = confusion_matrix(y_test, y_pred_test, labels=LABELS)
    cm_test_csv = os.path.join(RESULTS_DIR, 'confusion_test_ensemble.csv')
    cm_test_png = os.path.join(RESULTS_DIR, 'confusion_test_ensemble.png')
    save_confusion_matrix(cm_test, cm_test_csv, cm_test_png, labels=LABEL_NAMES, title='Test Ensemble')
    print(f"Saved test ensemble confusion to: {cm_test_csv}, {cm_test_png}")

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

    # Test row (ไม่ใส่ loss ตามสไตล์เดิม)
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

    # ---------------- Save CSV ----------------
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
                f"{row[4]} & {row[5]} & {row[6]} & {row[7]}\\\\\n"
            )
        f.write("\\hline\n")
        f.write("\\end{tabular}\n")
        f.write("\\caption{LFCC ResNet34 4-fold cross-validation using setA--setD as folds and Test as evaluation set.}\n")
        f.write("\\label{tab:lfcc_resnet34_4fold}\n")
        f.write("\\end{table}\n")
    print(f"Saved LaTeX table to: {RESULTS_TEX}")


if __name__ == "__main__":
    main()
