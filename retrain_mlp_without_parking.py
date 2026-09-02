
"""
required_car_parking_spaces 제거 후 MLP 재검증 + 최종 재학습.

- 기존 공통 분할(common_data_split.joblib)을 그대로 사용
- 기존 최종 선정 구조/하이퍼파라미터를 그대로 사용
- Broad/Focused 290회 탐색은 다시 하지 않음
- 기존 final 파일은 덮어쓰지 않고 model_dl/retrain_no_parking/에 별도 저장

실행:
    python retrain_mlp_without_parking.py
"""

from pathlib import Path
from datetime import datetime, timezone
import json
import random
import time

import joblib
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import OneHotEncoder, StandardScaler
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    roc_auc_score,
    confusion_matrix,
)

from tensorflow.keras import Sequential
from tensorflow.keras.layers import Input, Dense, BatchNormalization, Dropout
from tensorflow.keras.regularizers import l2


ROOT = Path(__file__).resolve().parent
OUT = ROOT / "model_dl/retrain_no_parking"
OUT.mkdir(parents=True, exist_ok=True)

SPLIT_PATH = ROOT / "model_dl/comparison/common_data_split.joblib"

DROP_FEATURE = "required_car_parking_spaces"

SEED = 42
EPOCHS = 107
BATCH_SIZE = 512
LEARNING_RATE = 0.0005
L2_VALUE = 0.0005
DROPOUT = 0.4


def set_seed(seed=SEED):
    random.seed(seed)
    np.random.seed(seed)
    tf.keras.utils.set_random_seed(seed)
    try:
        tf.config.experimental.enable_op_determinism()
    except Exception:
        pass


def build_preprocessor(categorical_cols, numeric_cols):
    categorical_transformer = Pipeline([
        (
            "onehot",
            OneHotEncoder(
                handle_unknown="ignore",
                sparse_output=False,
            ),
        )
    ])

    numeric_transformer = Pipeline([
        ("scaler", StandardScaler())
    ])

    return ColumnTransformer([
        ("num", numeric_transformer, numeric_cols),
        ("cat", categorical_transformer, categorical_cols),
    ])


def build_model(n_features):
    model = Sequential([
        Input(shape=(n_features,)),

        Dense(
            256,
            activation="relu",
            kernel_regularizer=l2(L2_VALUE),
        ),
        BatchNormalization(),
        Dropout(DROPOUT),

        Dense(
            128,
            activation="relu",
            kernel_regularizer=l2(L2_VALUE),
        ),
        BatchNormalization(),
        Dropout(DROPOUT),

        Dense(
            64,
            activation="relu",
            kernel_regularizer=l2(L2_VALUE),
        ),
        BatchNormalization(),
        Dropout(DROPOUT),

        Dense(1, activation="sigmoid"),
    ])

    model.compile(
        optimizer=tf.keras.optimizers.Adam(
            learning_rate=LEARNING_RATE
        ),
        loss="binary_crossentropy",
        metrics=[
            "accuracy",
            tf.keras.metrics.AUC(name="roc_auc"),
            tf.keras.metrics.Precision(name="precision"),
            tf.keras.metrics.Recall(name="recall"),
        ],
    )

    return model


def calc_metrics(y_true, proba, threshold):
    pred = (proba >= threshold).astype(int)
    tn, fp, fn, tp = confusion_matrix(y_true, pred).ravel()

    return {
        "threshold": float(threshold),
        "accuracy": float(accuracy_score(y_true, pred)),
        "precision": float(precision_score(y_true, pred, zero_division=0)),
        "recall": float(recall_score(y_true, pred, zero_division=0)),
        "f1": float(f1_score(y_true, pred, zero_division=0)),
        "roc_auc": float(roc_auc_score(y_true, proba)),
        "tn": int(tn),
        "fp": int(fp),
        "fn": int(fn),
        "tp": int(tp),
    }


def print_metrics(title, m):
    print(f"\n[{title}]")
    print(f"Threshold : {m['threshold']:.3f}")
    print(f"Accuracy  : {m['accuracy']:.6f}")
    print(f"Precision : {m['precision']:.6f}")
    print(f"Recall    : {m['recall']:.6f}")
    print(f"F1        : {m['f1']:.6f}")
    print(f"ROC-AUC   : {m['roc_auc']:.6f}")
    print(f"CM        : [[{m['tn']}, {m['fp']}], [{m['fn']}, {m['tp']}]]")


def main():
    if not SPLIT_PATH.is_file():
        raise FileNotFoundError(f"공통 분할 파일 없음: {SPLIT_PATH}")

    set_seed()

    split = joblib.load(SPLIT_PATH)

    X_train = split["X_train"].copy()
    X_test = split["X_test"].copy()
    X_subtrain = split["X_subtrain"].copy()
    X_val = split["X_val"].copy()

    y_train = split["y_train"].copy()
    y_test = split["y_test"].copy()
    y_subtrain = split["y_subtrain"].copy()
    y_val = split["y_val"].copy()

    categorical_cols = list(split["categorical_cols"])
    numeric_cols = list(split["numeric_cols"])

    print("=" * 72)
    print("주차 공간 변수 제거 MLP 재검증")
    print("=" * 72)

    # 원인 재확인
    parking_check = pd.DataFrame({
        "parking": split["X_train"][DROP_FEATURE],
        "is_canceled": y_train,
    })
    parking_summary = (
        parking_check
        .groupby("parking")
        .agg(
            n_samples=("is_canceled", "size"),
            canceled=("is_canceled", "sum"),
            cancel_rate=("is_canceled", "mean"),
        )
        .reset_index()
    )
    parking_summary["cancel_rate_pct"] = parking_summary["cancel_rate"] * 100
    parking_summary.to_csv(
        OUT / "parking_cancel_rate_before_drop.csv",
        index=False,
        encoding="utf-8-sig",
    )

    print("\n[주차 공간별 Train 취소율]")
    print(parking_summary.to_string(index=False))

    # Feature 제거
    for name, frame in {
        "X_train": X_train,
        "X_test": X_test,
        "X_subtrain": X_subtrain,
        "X_val": X_val,
    }.items():
        if DROP_FEATURE not in frame.columns:
            raise ValueError(f"{name}에 {DROP_FEATURE} 없음")

    if DROP_FEATURE not in numeric_cols:
        raise ValueError(f"numeric_cols에 {DROP_FEATURE} 없음")

    X_train = X_train.drop(columns=[DROP_FEATURE])
    X_test = X_test.drop(columns=[DROP_FEATURE])
    X_subtrain = X_subtrain.drop(columns=[DROP_FEATURE])
    X_val = X_val.drop(columns=[DROP_FEATURE])
    numeric_cols.remove(DROP_FEATURE)

    print("\n제거 Feature:", DROP_FEATURE)
    print("원본 입력 의미 변수: 20개 → 19개")

    # ---------------------------------------------------------
    # 1) Validation 재검증
    # ---------------------------------------------------------
    print("\n1/3 Validation 전처리 + 107 Epoch 학습")

    val_preprocessor = build_preprocessor(
        categorical_cols,
        numeric_cols,
    )

    X_subtrain_t = val_preprocessor.fit_transform(
        X_subtrain
    ).astype("float32")
    X_val_t = val_preprocessor.transform(
        X_val
    ).astype("float32")

    print("변환 후 입력 Feature 수:", X_subtrain_t.shape[1])

    y_subtrain_np = y_subtrain.to_numpy(dtype="float32")
    y_val_np = y_val.to_numpy(dtype="int32")

    set_seed()
    val_model = build_model(X_subtrain_t.shape[1])

    start = time.perf_counter()

    val_history = val_model.fit(
        X_subtrain_t,
        y_subtrain_np,
        validation_data=(X_val_t, y_val_np),
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1,
    )

    val_train_minutes = (time.perf_counter() - start) / 60

    val_proba = val_model.predict(
        X_val_t,
        batch_size=2048,
        verbose=0,
    ).ravel()

    # 임계값 재선정
    thresholds = np.round(
        np.arange(0.200, 0.801, 0.001),
        3,
    )

    rows = []
    for threshold in thresholds:
        rows.append(calc_metrics(y_val_np, val_proba, threshold))

    threshold_df = pd.DataFrame(rows)

    max_acc = threshold_df["accuracy"].max()
    candidates = threshold_df.loc[
        np.isclose(threshold_df["accuracy"], max_acc)
    ].copy()

    # 기존 방식: 정확도 동률이면 0.5와 가까운 값
    candidates["distance_from_05"] = (
        candidates["threshold"] - 0.5
    ).abs()

    best = (
        candidates
        .sort_values(
            ["distance_from_05", "f1"],
            ascending=[True, False],
        )
        .iloc[0]
    )

    optimized_threshold = float(best["threshold"])

    val_05 = calc_metrics(y_val_np, val_proba, 0.5)
    val_opt = calc_metrics(
        y_val_np,
        val_proba,
        optimized_threshold,
    )

    print_metrics("Validation @ 0.500", val_05)
    print_metrics(
        f"Validation @ 최적 {optimized_threshold:.3f}",
        val_opt,
    )
    print(f"\nValidation 학습시간: {val_train_minutes:.2f}분")

    # 저장
    joblib.dump(
        val_preprocessor,
        OUT / "mlp_validation_preprocessor_no_parking.joblib",
    )
    val_model.save(
        OUT / "selected_mlp_validation_epoch107_no_parking.keras"
    )

    pd.DataFrame({
        "original_index": X_val.index,
        "y_true": y_val_np,
        "cancel_probability": val_proba,
    }).to_csv(
        OUT / "mlp_validation_predictions_no_parking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    threshold_df.to_csv(
        OUT / "mlp_threshold_results_no_parking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    history_df = pd.DataFrame(val_history.history)
    history_df.insert(0, "epoch", np.arange(1, len(history_df) + 1))
    history_df.to_csv(
        OUT / "mlp_validation_history_no_parking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    # ---------------------------------------------------------
    # 2) 전체 Train 최종 재학습
    # ---------------------------------------------------------
    print("\n2/3 전체 Train 전처리 + 107 Epoch 최종 학습")

    final_preprocessor = build_preprocessor(
        categorical_cols,
        numeric_cols,
    )

    X_train_t = final_preprocessor.fit_transform(
        X_train
    ).astype("float32")
    X_test_t = final_preprocessor.transform(
        X_test
    ).astype("float32")

    print("최종 변환 후 입력 Feature 수:", X_train_t.shape[1])

    y_train_np = y_train.to_numpy(dtype="float32")
    y_test_np = y_test.to_numpy(dtype="int32")

    set_seed()
    final_model = build_model(X_train_t.shape[1])

    start = time.perf_counter()

    final_history = final_model.fit(
        X_train_t,
        y_train_np,
        epochs=EPOCHS,
        batch_size=BATCH_SIZE,
        shuffle=True,
        verbose=1,
    )

    final_train_minutes = (time.perf_counter() - start) / 60

    test_proba = final_model.predict(
        X_test_t,
        batch_size=2048,
        verbose=0,
    ).ravel()

    test_05 = calc_metrics(y_test_np, test_proba, 0.5)
    test_opt = calc_metrics(
        y_test_np,
        test_proba,
        optimized_threshold,
    )

    print_metrics("Common Test @ 0.500", test_05)
    print_metrics(
        f"Common Test @ Validation 최적 {optimized_threshold:.3f}",
        test_opt,
    )
    print(f"\n최종 학습시간: {final_train_minutes:.2f}분")

    # ---------------------------------------------------------
    # 3) 산출물 저장
    # ---------------------------------------------------------
    print("\n3/3 별도 폴더에 저장")

    joblib.dump(
        final_preprocessor,
        OUT / "final_mlp_preprocessor_no_parking.joblib",
    )
    final_model.save(
        OUT / "final_mlp_no_parking.keras"
    )

    pd.DataFrame({
        "original_index": X_test.index,
        "y_true": y_test_np,
        "mlp_probability": test_proba,
    }).to_csv(
        OUT / "final_mlp_test_predictions_no_parking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    final_hist_df = pd.DataFrame(final_history.history)
    final_hist_df.insert(
        0,
        "epoch",
        np.arange(1, len(final_hist_df) + 1),
    )
    final_hist_df.to_csv(
        OUT / "final_mlp_history_no_parking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    metrics_df = pd.DataFrame([
        {"data_split": "Validation", **val_05},
        {"data_split": "Validation", **val_opt},
        {"data_split": "Common Test", **test_05},
        {"data_split": "Common Test", **test_opt},
    ])
    metrics_df.to_csv(
        OUT / "mlp_metrics_no_parking.csv",
        index=False,
        encoding="utf-8-sig",
    )

    config = {
        "created_at": datetime.now(timezone.utc).isoformat(),
        "experiment": "drop_required_car_parking_spaces",
        "drop_feature": DROP_FEATURE,
        "selection_note": (
            "기존 290회 탐색에서 선정한 최종 구조/하이퍼파라미터를 "
            "그대로 사용한 재검증이며, 제거 후 재탐색은 수행하지 않음"
        ),
        "units": [256, 128, 64],
        "optimizer": "Adam",
        "learning_rate": LEARNING_RATE,
        "dropout": DROPOUT,
        "l2": L2_VALUE,
        "batch_norm": True,
        "epochs": EPOCHS,
        "batch_size": BATCH_SIZE,
        "seed": SEED,
        "validation_optimized_threshold": optimized_threshold,
        "input_features_after_transform": int(X_train_t.shape[1]),
        "raw_features": list(X_train.columns),
        "categorical_cols": categorical_cols,
        "numeric_cols": numeric_cols,
        "validation_metrics_05": val_05,
        "validation_metrics_optimized": val_opt,
        "test_metrics_05": test_05,
        "test_metrics_optimized": test_opt,
    }

    (OUT / "final_mlp_config_no_parking.json").write_text(
        json.dumps(config, ensure_ascii=False, indent=2),
        encoding="utf-8",
    )

    print(f"\n저장 위치: {OUT}")
    print("\n*** 중요 ***")
    print("기존 model_dl/final 파일은 덮어쓰지 않았습니다.")
    print(
        "결과를 확인한 뒤 성능/민감도가 괜찮으면 "
        "그때 배포용 final 파일로 승격합니다."
    )


if __name__ == "__main__":
    main()
