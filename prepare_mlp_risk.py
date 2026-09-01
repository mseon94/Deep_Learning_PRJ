"""프로젝트 루트에서 한 번 실행: python prepare_mlp_risk.py

모델/전처리기를 재학습하지 않습니다. 저장된 분할과 예측을 대조한 뒤
MLP Validation의 예상 손실 50/75% 분위수를 고정하고 Test에 적용합니다.
생성/교체 대상은 model_dl/risk/의 전용 자료 세 파일뿐입니다.
"""
from pathlib import Path
from datetime import datetime, timezone
import hashlib
import json
import os
import tempfile

import joblib
import numpy as np
import pandas as pd

ROOT = Path(__file__).resolve().parent
LEVELS = ["LOW", "MEDIUM", "HIGH"]


def require(condition, message):
    if not condition:
        raise ValueError(message)


def probabilities(model, processor, frame):
    values = processor.transform(frame)
    if hasattr(values, "toarray"):
        values = values.toarray()
    result = model.predict(np.asarray(values, dtype="float32"), batch_size=1024, verbose=0).ravel()
    require(len(result) == len(frame), "예측 건수가 다릅니다.")
    require(np.isfinite(result).all() and ((result >= 0) & (result <= 1)).all(), "유효하지 않은 예측확률입니다.")
    return result


def risk_summary(frame, y, probability, q50, q75, split_name):
    amounts = frame["adr"].to_numpy(dtype=float) * frame["total_nights"].to_numpy(dtype=float)
    require(np.isfinite(amounts).all() and (amounts >= 0).all(), "예약 금액 지표가 유효하지 않습니다.")
    loss = amounts * probability
    levels = np.where(loss < q50, "LOW", np.where(loss < q75, "MEDIUM", "HIGH"))
    actual = np.asarray(y, dtype=float)
    require(len(actual) == len(frame) == len(probability), "위험등급 집계 길이가 다릅니다.")
    data = pd.DataFrame({"risk_level": levels, "actual": actual, "probability": probability,
                         "loss": loss, "canceled_amount": actual * amounts})
    result = data.groupby("risk_level").agg(
        n_samples=("actual", "size"), canceled=("actual", "sum"),
        actual_cancel_rate=("actual", "mean"),
        mean_probability=("probability", "mean"),
        mean_expected_loss=("loss", "mean"),
        canceled_booking_amount=("canceled_amount", "sum")
    ).reindex(LEVELS)
    result["n_samples"] = result["n_samples"].fillna(0).astype(int)
    result["canceled"] = result["canceled"].fillna(0).astype(int)
    result["canceled_booking_amount"] = result["canceled_booking_amount"].fillna(0)
    require(actual.sum() > 0 and data["canceled_amount"].sum() > 0, "취소 건수·금액 합계가 0입니다.")
    result["booking_share"] = result["n_samples"] / len(data)
    result["cancellation_capture_rate"] = result["canceled"] / actual.sum()
    result["canceled_amount_capture_rate"] = result["canceled_booking_amount"] / data["canceled_amount"].sum()
    result["data_split"] = split_name
    return result.reset_index()


def sha256(path):
    digest = hashlib.sha256()
    with path.open("rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def main():
    paths = {
        "split": ROOT / "model_dl/comparison/common_data_split.joblib",
        "val_model": ROOT / "model_dl/validation/selected_mlp_validation_epoch107.keras",
        "val_processor": ROOT / "model_dl/search/mlp_search_preprocessor.joblib",
        "val_predictions": ROOT / "model_dl/validation/mlp_validation_predictions.csv",
        "final_model": ROOT / "model_dl/final/final_mlp.keras",
        "final_processor": ROOT / "model_dl/final/final_mlp_preprocessor.joblib",
        "test_predictions": ROOT / "model_dl/comparison/verified_test/rf_mlp_common_test_predictions.csv",
        "capture": ROOT / "model_dl/comparison/capture_analysis/mlp_validation_capture.csv",
    }
    missing = [str(path) for path in paths.values() if not path.is_file()]
    if missing:
        raise FileNotFoundError("필요한 파일이 없습니다:\n" + "\n".join(missing))

    import tensorflow as tf
    split = joblib.load(paths["split"])
    val_x, val_y = split["X_val"], split["y_val"]
    test_x, test_y = split["X_test"], split["y_test"]
    for x, y in [(val_x, val_y), (test_x, test_y)]:
        require(x.index.is_unique and y.index.is_unique and x.index.equals(y.index), "분할 내 예약 인덱스·정답 순서가 다릅니다.")
        require(pd.Series(y).isin([0, 1]).all(), "정답은 0/1이어야 합니다.")
    require(not val_x.index.isin(test_x.index).any(), "Validation과 Test가 겹칩니다.")
    require(not split["X_subtrain"].index.isin(val_x.index).any(), "Sub-train과 Validation이 겹칩니다.")

    print("1/3 저장된 Validation 모델과 예측확률 대조 중 (재학습 없음)")
    val_model = tf.keras.models.load_model(paths["val_model"], compile=False)
    val_processor = joblib.load(paths["val_processor"])
    val_p = probabilities(val_model, val_processor, val_x)
    saved_val = pd.read_csv(paths["val_predictions"])
    require(len(saved_val) == len(val_y), "Validation 저장 건수가 다릅니다.")
    require(np.array_equal(saved_val["y_true"].to_numpy(), val_y.to_numpy()), "Validation 정답 순서가 다릅니다.")
    require(np.allclose(saved_val["cancel_probability"], val_p, rtol=1e-5, atol=1e-6), "Validation 확률 대조 실패: 전처리기와 모델의 조합을 확인하세요. 자료를 저장하지 않습니다.")
    # 재실행 시 부동소수점 차이를 피하도록, 대조를 통과한 저장 확률을 사용
    val_p = saved_val["cancel_probability"].to_numpy(dtype=float)
    val_loss = val_p * val_x["adr"].to_numpy(dtype=float) * val_x["total_nights"].to_numpy(dtype=float)
    capture = pd.read_csv(paths["capture"])
    medium = capture.loc[np.isclose(capture["target_review_share"], .50)]
    high = capture.loc[np.isclose(capture["target_review_share"], .25)]
    require(len(medium) == len(high) == 1, "포착률 분석표에서 상위 25%·50% 행을 확인하세요.")
    q50 = float(medium.iloc[0]["expected_loss_threshold"])
    q75 = float(high.iloc[0]["expected_loss_threshold"])
    require(np.allclose([q50, q75], np.quantile(val_loss, [.5,.75]), rtol=1e-9, atol=1e-8),
            "포착률 분석의 경계값이 현재 Validation 자료와 일치하지 않습니다.")
    require(np.isfinite([q50, q75]).all() and 0 < q50 < q75, "분위수가 0 또는 중복입니다. 점수 산식을 확인해야 합니다.")

    print("2/3 공통 Test 인덱스·정답·최종 MLP 확률 대조 중")
    saved_test = pd.read_csv(paths["test_predictions"])
    require(saved_test["original_index"].is_unique, "Test 예약 인덱스가 중복입니다.")
    saved_test = saved_test.set_index("original_index")
    require(set(saved_test.index) == set(test_x.index), "공통 Test 예약 인덱스가 다릅니다.")
    saved_test = saved_test.loc[test_x.index]
    require(np.array_equal(saved_test["y_true"].to_numpy(), test_y.to_numpy()), "Test 정답이 다릅니다.")
    final_model = tf.keras.models.load_model(paths["final_model"], compile=False)
    final_processor = joblib.load(paths["final_processor"])
    test_p = probabilities(final_model, final_processor, test_x)
    require(np.allclose(saved_test["mlp_probability"], test_p, rtol=1e-5, atol=1e-6), "최종 MLP Test 확률이 기존 공통 Test 자료와 다릅니다.")
    test_p = saved_test["mlp_probability"].to_numpy(dtype=float)

    val_summary = risk_summary(val_x, val_y, val_p, q50, q75, "Validation")
    test_summary = risk_summary(test_x, test_y, test_p, q50, q75, "Common Test")
    config = {
        "schema_version": 2, "model": "MLP", "selection_split": "Validation",
        "method": "expected_loss_quantiles", "q50": float(q50), "q75": float(q75),
        "validation_n": len(val_x), "test_n": len(test_x),
        "formula": "cancel_probability * adr * total_nights",
        "boundaries": "LOW: loss < q50; MEDIUM: q50 <= loss < q75; HIGH: loss >= q75",
        "transfer_note": "Validation 모델로 정한 경계를 최종 MLP에 그대로 적용; Test에서 경계 재조정하지 않음",
        "created_at": datetime.now(timezone.utc).isoformat(),
        "source_sha256": {name: sha256(path) for name, path in paths.items()},
    }
    output = ROOT / "model_dl/risk"
    output.mkdir(parents=True, exist_ok=True)
    # 대조가 전부 통과한 뒤에만 전용 결과를 기록. 기존 모델·원본 CSV는 변경하지 않음.
    with tempfile.TemporaryDirectory(prefix="risk_export_", dir=output) as temp:
        temp = Path(temp)
        val_summary.to_csv(temp / "mlp_risk_validation_summary.csv", index=False, encoding="utf-8-sig")
        test_summary.to_csv(temp / "mlp_risk_test_summary.csv", index=False, encoding="utf-8-sig")
        config["summary_sha256"] = {name: sha256(temp / name) for name in
            ["mlp_risk_validation_summary.csv", "mlp_risk_test_summary.csv"]}
        (temp / "mlp_risk_thresholds.json").write_text(json.dumps(config, ensure_ascii=False, indent=2), encoding="utf-8")
        for name in ["mlp_risk_validation_summary.csv", "mlp_risk_test_summary.csv", "mlp_risk_thresholds.json"]:
            os.replace(temp / name, output / name)
    print(f"3/3 저장 완료: {output}\nMLP q50={q50:.6f}, q75={q75:.6f}")
    print(test_summary.to_string(index=False))


if __name__ == "__main__":
    main()
