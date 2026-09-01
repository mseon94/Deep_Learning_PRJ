"""Validation 관리 비율-취소 예약금액 포착률 분석.

실행: 프로젝트 루트에 이 파일을 놓고 노트북에서 %run ./analyze_mlp_capture.py
모델/전처리기 학습 없음, Test 사용 없음, 서비스 등급 경계 변경 없음.
결과는 model_dl/comparison/capture_analysis/의 CSV와 PNG 두 개에 저장됩니다.
재실행하면 이 분석 결과 두 파일만 교체합니다.
"""
from pathlib import Path
import math
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parent


def require(condition, message):
    if not condition:
        raise ValueError(message)


def capture_analysis(frame, labels, probability):
    """동일 예상 손실 점수의 예약은 함께 선택하여 동점 순서 영향을 제거."""
    y = np.asarray(labels, dtype=float).reshape(-1)
    p = np.asarray(probability, dtype=float).reshape(-1)
    amount = (frame["adr"].to_numpy(dtype=float)
              * frame["total_nights"].to_numpy(dtype=float))
    require(len(frame) > 0 and len(frame) == len(y) == len(p), "데이터 길이가 다릅니다.")
    require(np.isin(y, [0, 1]).all(), "정답은 0/1이어야 합니다.")
    require(np.isfinite(p).all() and ((p >= 0) & (p <= 1)).all(), "확률 범위를 확인하세요.")
    require(np.isfinite(amount).all() and (amount >= 0).all(), "예약금액 범위를 확인하세요.")
    data = pd.DataFrame({
        "risk_score": p * amount,
        "canceled": y,
        "canceled_booking_amount": y * amount,
    })
    total_canceled = int(y.sum())
    total_canceled_amount = float((y * amount).sum())
    require(total_canceled > 0 and total_canceled_amount > 0,
            "취소 건수 또는 취소 예약금액 합계가 0이므로 포착률을 계산할 수 없습니다.")
    rows = []
    # 목표 상위 비율과 실제 비율은 동점 때문에 다를 수 있음
    for target in [0.05, 0.10, 0.15, 0.20, 0.25, 0.30, 0.40, 0.50, 0.60, 0.70, 0.80, 0.90, 1.0]:
        threshold = float(np.quantile(data["risk_score"], 1 - target))
        selected = data.loc[data["risk_score"] >= threshold]
        n = len(selected)
        canceled = int(selected["canceled"].sum())
        captured_amount = float(selected["canceled_booking_amount"].sum())
        actual_share = n / len(data)
        coverage = captured_amount / total_canceled_amount
        rows.append({
            "data_split": "Validation", "n_total": len(data),
            "target_review_share": target, "actual_review_share": actual_share,
            "expected_loss_threshold": threshold, "n_review": n,
            "n_canceled_in_review": canceled, "n_not_canceled_in_review": n-canceled,
            "cancel_rate_in_review": canceled / n,
            "cancellation_capture_rate": canceled / total_canceled,
            "canceled_amount_capture_rate": coverage,
            "amount_capture_lift": coverage / actual_share,
            "captured_canceled_booking_amount": captured_amount,
            "total_canceled_booking_amount": total_canceled_amount,
            "reference": ("q75: HIGH candidate" if math.isclose(target, .25)
                          else "q50: MEDIUM+HIGH candidate" if math.isclose(target, .5) else ""),
        })
    comparison = pd.DataFrame(rows)
    # 누적 포착률 곡선. 동점 예약을 하나의 그룹으로 처리합니다.
    groups = data.groupby("risk_score", sort=True).agg(
        n=("canceled", "size"), canceled=("canceled", "sum"),
        amount=("canceled_booking_amount", "sum")
    ).sort_index(ascending=False)
    curve = pd.DataFrame({
        "review_share": np.r_[0, groups["n"].cumsum().to_numpy() / len(data)],
        "count_capture": np.r_[0, groups["canceled"].cumsum().to_numpy() / total_canceled],
        "amount_capture": np.r_[0, groups["amount"].cumsum().to_numpy() / total_canceled_amount],
    })
    return comparison, curve


def make_plot(comparison, curve):
    fig, ax = plt.subplots(figsize=(12, 6))
    ax.plot(curve["review_share"] * 100, curve["amount_capture"] * 100,
            color="#E87924", linewidth=2.5, label="Canceled booking amount captured")
    ax.plot(curve["review_share"] * 100, curve["count_capture"] * 100,
            color="#3677AD", linewidth=2, label="Canceled bookings captured")
    ax.plot([0, 100], [0, 100], "--", color="gray", label="Random selection (expectation)")
    for target, label, offset in [(.25, "q75 / HIGH candidate", 10), (.5, "q50 / MEDIUM + HIGH candidate", -42)]:
        row = comparison.loc[np.isclose(comparison["target_review_share"], target)].iloc[0]
        x, y = row["actual_review_share"]*100, row["canceled_amount_capture_rate"]*100
        ax.axvline(x, color="gray", linestyle=":", alpha=.7)
        ax.scatter([x], [y], color="#E87924", zorder=3)
        ax.annotate(f"{label}\nReview {x:.1f}% / Amount {y:.1f}%",
                    (x, y), xytext=(8, offset), textcoords="offset points", fontsize=9)
    ax.set(xlim=(0,100), ylim=(0,105), xlabel="Bookings selected for review (%)",
           ylabel="Share of all canceled bookings / canceled amount (%)",
           title="MLP Validation — Review Workload vs Capture\nRanked by predicted cancellation probability × booking amount")
    ax.grid(alpha=.2)
    ax.legend(loc="lower right")
    fig.tight_layout()
    return fig


def main():
    import joblib
    import tensorflow as tf
    paths = {
        "split": ROOT / "model_dl/comparison/common_data_split.joblib",
        "model": ROOT / "model_dl/validation/selected_mlp_validation_epoch107.keras",
        "processor": ROOT / "model_dl/search/mlp_search_preprocessor.joblib",
        "predictions": ROOT / "model_dl/validation/mlp_validation_predictions.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("필요한 파일이 없습니다:\n" + "\n".join(missing))
    split = joblib.load(paths["split"])
    x, y = split["X_val"], split["y_val"]
    require(x.index.is_unique and y.index.is_unique and x.index.equals(y.index),
            "Validation 예약 인덱스·정답 순서가 일치하지 않습니다.")
    require(not x.index.isin(split["X_subtrain"].index).any(), "Sub-train과 Validation이 겹칩니다.")
    saved = pd.read_csv(paths["predictions"])
    require(len(saved) == len(y) and np.array_equal(saved["y_true"].to_numpy(), y.to_numpy()),
            "저장된 Validation 정답이 분할 파일과 다릅니다.")

    print("저장된 Validation 모델로 예측 순서·확률을 대조합니다. 재학습하지 않습니다.")
    processor = joblib.load(paths["processor"])
    model = tf.keras.models.load_model(paths["model"], compile=False)
    transformed = processor.transform(x)
    if hasattr(transformed, "toarray"):
        transformed = transformed.toarray()
    current = model.predict(np.asarray(transformed, dtype="float32"), batch_size=1024, verbose=0).ravel()
    stored = saved["cancel_probability"].to_numpy(dtype=float)
    require(len(current) == len(stored) and np.allclose(current, stored, rtol=1e-5, atol=1e-6),
            "확률 대조 실패: Validation 모델·전처리기·분할 조합을 확인하세요. 결과를 저장하지 않습니다.")
    print(f"대조 통과 | Validation {len(x):,}건 | 최대 확률 차이 {np.max(np.abs(current-stored)):.9g}")
    comparison, curve = capture_analysis(x, y, stored)
    output = ROOT / "model_dl/comparison/capture_analysis"
    output.mkdir(parents=True, exist_ok=True)
    comparison.to_csv(output / "mlp_validation_capture.csv", index=False, encoding="utf-8-sig")
    fig = make_plot(comparison, curve)
    fig.savefig(output / "mlp_validation_capture.png", dpi=200, bbox_inches="tight")
    display_table = comparison[[
        "target_review_share", "actual_review_share", "n_review", "expected_loss_threshold",
        "cancel_rate_in_review", "cancellation_capture_rate", "canceled_amount_capture_rate", "amount_capture_lift"
    ]].copy()
    percentage_columns = ["target_review_share", "actual_review_share", "cancel_rate_in_review",
                          "cancellation_capture_rate", "canceled_amount_capture_rate"]
    display_table[percentage_columns] *= 100
    display_table.columns = ["목표 상위 비율(%)", "실제 관리 비율(%)", "관리 건수", "예상 손실 경계값",
                             "관리 대상 실제 취소율(%)", "취소 건수 포착률(%)", "취소 예약금액 포착률(%)", "금액 포착 배율"]
    try:
        from IPython.display import display
        display(display_table.round(2))
    except ImportError:
        print(display_table.round(2).to_string(index=False))
    print("기존 q75는 상위 약 25%, q50은 상위 약 50% 관리에 해당합니다.")
    print("동점 예약은 함께 선택하므로 목표 비율과 실제 비율은 다를 수 있습니다.")
    print("주의: 취소 예약금액 포착률이지 실제 손실 절감률이 아닙니다. 관리로 취소가 예방된다고 가정하지 않습니다.")
    print(f"저장 완료: {output}\n서비스 경계값·모델 파일은 변경하지 않았습니다.")
    plt.show()
    plt.close(fig)


if __name__ == "__main__":
    main()
