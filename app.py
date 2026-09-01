import os
from pathlib import Path
import json
import hashlib
import numpy as np
import joblib
import pandas as pd
import matplotlib.pyplot as plt
import streamlit as st
import tensorflow as tf
from matplotlib import font_manager

ROOT = Path(__file__).resolve().parent

# 페이지 설정은 다른 Streamlit 명령보다 먼저 실행
st.set_page_config(
    page_title="호텔 예약 취소 예측",
    page_icon="🏨",
    layout="wide"
)

# 로컬 Windows와 Streamlit Linux 배포 환경 모두 대응
if os.name == "nt":
    plt.rcParams["font.family"] = "Malgun Gothic"
else:
    font_path = "/usr/share/fonts/truetype/nanum/NanumGothic.ttf"
    if os.path.exists(font_path):
        font_manager.fontManager.addfont(font_path)
        font_name = font_manager.FontProperties(
            fname=font_path
        ).get_name()
        plt.rcParams["font.family"] = font_name

plt.rcParams["axes.unicode_minus"] = False

# CSS
st.markdown("""
<style>

/* 전체 화면 여백 */
.block-container {
    padding-top: 2rem;
    padding-bottom: 3rem;
    max-width: 1400px;
}

/* 제목 */
.main-title {
    font-size: 2.2rem;
    font-weight: 700;
    margin-bottom: 0.3rem;
}

/* 설명 */
.main-description {
    font-size: 1rem;
    color: #6b7280;
    margin-bottom: 2rem;
}

/* 결과 위험등급 */
.risk-badge {
    display: inline-block;
    padding: 10px 18px;
    border-radius: 999px;
    font-size: 1.2rem;
    font-weight: 700;
    text-align: center;
    min-width: 90px;
}

.risk-badge.low {
    background-color: #dcfce7;
    color: #15803d;
    border: 1px solid #bbf7d0;
}

.risk-badge.medium {
    background-color: #fef3c7;
    color: #b45309;
    border: 1px solid #fde68a;
}

.risk-badge.high {
    background-color: #fee2e2;
    color: #b91c1c;
    border: 1px solid #fecaca;
}

.result-label {
    font-size: 0.9rem;
    color: #6b7280;
    margin-bottom: 0.2rem;
}

.result-value {
    font-size: 2rem;
    font-weight: 700;
    line-height: 1.1;
}

.section-card-title {
    font-size: 1.1rem;
    font-weight: 700;
    margin-bottom: 0.8rem;
}

.centered-result-label {
    color: #6b7280;
    font-size: 0.88rem;
    line-height: 1.35;
    margin-bottom: 0.35rem;
    text-align: center;
}

.priority-result-value {
    font-size: 1.65rem;
    font-weight: 700;
    line-height: 1.15;
    text-align: center;
    white-space: nowrap;
}

.amount-result-value {
    font-size: 1.8rem;
    font-weight: 400;
    line-height: 1.2;
    text-align: center;
    white-space: nowrap;
    padding-bottom: 24px;
}

.risk-badge.centered {
    display: block;
    margin-left: auto;
    margin-right: auto;
    width: fit-content;
}

.st-key-priority_progress {
    margin-top: 12px;
    margin-bottom: -12px;
}

</style>
""", unsafe_allow_html=True)

st.markdown(
    '<div class="main-title">호텔 예약 취소 위험 예측</div>',
    unsafe_allow_html=True
)

st.markdown(
    """
    <div class="main-description">
        예약 정보를 입력하면 최종 MLP 모델이 취소 가능성과 예상 손실 지표를 계산합니다.
    </div>
    """,
    unsafe_allow_html=True
)


# 자료 불러오기
@st.cache_resource
def load_mlp_objects(model_stamp, processor_stamp):
    model = tf.keras.models.load_model(ROOT / "model_dl/final/final_mlp.keras", compile=False)
    preprocessor = joblib.load(ROOT / "model_dl/final/final_mlp_preprocessor.joblib")

    return model, preprocessor

try:
    model, preprocessor = load_mlp_objects(
        (ROOT / "model_dl/final/final_mlp.keras").stat().st_mtime_ns,
        (ROOT / "model_dl/final/final_mlp_preprocessor.joblib").stat().st_mtime_ns
    )
except Exception as error:
    st.error(f"최종 모델 또는 전처리기를 불러오지 못했습니다: {error}")
    st.stop()


# 위험등급 준비 자료는 현재 모델과 연결된 완성된 세트만 사용
@st.cache_data
def file_sha256(path, stamp):
    digest = hashlib.sha256()
    with open(path, "rb") as file:
        for block in iter(lambda: file.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def load_risk_bundle():
    path = ROOT / "model_dl/risk/mlp_risk_thresholds.json"
    if not path.is_file():
        return None, "위험등급 자료 없음: python prepare_mlp_risk.py"
    try:
        with path.open(encoding="utf-8-sig") as file:
            config = json.load(file)
        if config.get("schema_version") != 2 or config.get("selection_split") != "Validation":
            raise ValueError("위험등급 자료 형식 불일치")
        low, high = float(config["q50"]), float(config["q75"])
        if not np.isfinite([low, high]).all() or not 0 < low < high:
            raise ValueError("위험등급 경계값 오류")
        checks = [
            (ROOT / "model_dl/final/final_mlp.keras", config["source_sha256"]["final_model"]),
            (ROOT / "model_dl/final/final_mlp_preprocessor.joblib", config["source_sha256"]["final_processor"]),
        ]
        for name in ["mlp_risk_validation_summary.csv", "mlp_risk_test_summary.csv"]:
            checks.append((ROOT / "model_dl/risk" / name, config["summary_sha256"][name]))
        for file, expected in checks:
            if file_sha256(str(file), file.stat().st_mtime_ns) != expected:
                raise ValueError(f"위험등급 자료와 파일 버전 불일치: {file.name}")
        return config, None
    except (OSError, ValueError, KeyError, TypeError) as error:
        return None, f"위험등급 자료 오류: {error}"


risk_thresholds, risk_error = load_risk_bundle()

OPTIMIZED_THRESHOLD = 0.511
config_path = ROOT / "model_dl/final/final_mlp_config.json"
if config_path.is_file():
    with config_path.open(encoding="utf-8-sig") as file:
        final_config = json.load(file)
    OPTIMIZED_THRESHOLD = float(final_config.get("optimized_threshold", 0.511))
else:
    final_config = {}


@st.cache_data
def load_dashboard_data():
    rf_results = pd.DataFrame([
        ["Random Forest - Baseline", 0.994739, 0.877957,
         0.999503, 0.948090, 0.862035, 0.800566, 0.830164],
        ["Random Forest - Broad RandomizedSearchCV", 0.933579,
         0.872011, 0.987812, 0.946277, 0.862681, 0.780758,
         0.819678],
        ["Random Forest - Regularized Search", 0.875128, 0.859824,
         0.956380, 0.936323, 0.865985, 0.737974, 0.796871],
        ["Random Forest - Moderate Final", 0.898713, 0.866318,
         0.973155, 0.941980, 0.867047, 0.757329, 0.808482]
    ], columns=[
        "Model", "Train Accuracy", "Test Accuracy", "Train ROC-AUC",
        "Test ROC-AUC", "Precision", "Recall", "F1 Score"
    ])

    importance_df = pd.DataFrame([
        ["country_group", 0.144925],
        ["market_segment", 0.066426],
        ["customer_type", 0.057091],
        ["total_of_special_requests", 0.052880],
        ["lead_time", 0.051438],
        ["required_car_parking_spaces", 0.020054],
        ["previous_cancellations", 0.014743],
        ["adr", 0.014404],
        ["arrival_date_month", 0.009906],
        ["hotel", 0.009761]
    ], columns=["Feature", "Importance"])

    return rf_results, importance_df


rf_results, importance_df = load_dashboard_data()


# 범주형 변수 목록
categorical_cols = [
    "hotel",
    "arrival_date_month",
    "meal",
    "market_segment",
    "reserved_room_type",
    "customer_type",
    "country_group"
]

# 학습된 OneHotEncoder에서 실제 카테고리 가져오기
encoder = (preprocessor
    .named_transformers_["cat"]
    .named_steps["onehot"]
)

category_options = {
    col: list(categories)
    for col, categories in zip(
        categorical_cols,
        encoder.categories_
    )
}


# 사용자 입력 폼 ============================================================
simulator_tab, dashboard_tab = st.tabs(
    [
        "예약 취소 위험 시뮬레이터",
        "모델 인사이트 대시보드"
    ]
)
with simulator_tab:
    input_col, result_col = st.columns(
        [1.8, 1],
        gap="large"
    )

    with input_col:
        with st.form("prediction_form"):

            st.subheader("예약 기본 정보")

            col1, col2 = st.columns(2)

            with col1:
                hotel = st.selectbox(
                    "호텔 유형",
                    category_options["hotel"],
                    index=category_options["hotel"].index("City Hotel")
                )

                meal = st.selectbox(
                    "식사 유형",
                    category_options["meal"],
                    index=category_options["meal"].index("BB")
                )

                reserved_room_type = st.selectbox(
                    "예약 객실 유형",
                    category_options["reserved_room_type"],
                    index=category_options["reserved_room_type"].index("A")
                )

                customer_type = st.selectbox(
                    "고객 유형",
                    category_options["customer_type"],
                    index=category_options["customer_type"].index("Transient")
                )


            with col2:
                arrival_date_month = st.selectbox(
                    "도착 월",
                    category_options["arrival_date_month"],
                    index=category_options["arrival_date_month"].index("August")
                )

                market_segment = st.selectbox(
                    "예약 경로",
                    category_options["market_segment"],
                    index=category_options["market_segment"].index("Online TA")
                )

                country_group = st.selectbox(
                    "국가 그룹",
                    category_options["country_group"],
                    index=category_options["country_group"].index("PRT")
                )
                
                
            with st.container(border=True):
                st.subheader("예약 상세 정보")
                
                col1, col2 = st.columns(2)

                with col1:
                    lead_time = st.number_input(
                        "예약 후 체크인까지 남은 일수",
                        min_value=0,
                        value=69
                    )

                    total_nights = st.number_input(
                        "총 숙박일",
                        min_value=1,
                        value=3
                    )

                    required_car_parking_spaces = st.number_input(
                        "필요 주차 공간 수",
                        min_value=0,
                        value=0
                    )

                    total_of_special_requests = st.number_input(
                        "특별 요청 수",
                        min_value=0,
                        value=0
                    )                    

                with col2:
                    adr = st.number_input(
                        "ADR (평균 일일 객실 요금)",
                        min_value=0.0,
                        value=80.1
                    )

                    previous_bookings_not_canceled = st.number_input(
                        "이전 정상 예약 횟수",
                        min_value=0,
                        value=0
                    )
                    
                    previous_cancellations = st.number_input(
                        "이전 예약 취소 횟수",
                        min_value=0,
                        value=0
                    )                    

                
            with st.container(border=True):
                st.subheader("투숙객 정보")

                col1, col2, col3 = st.columns(3)

                with col1:
                    adults = st.number_input(
                        "성인",
                        min_value=1,
                        value=2
                    )

                with col2:
                    children = st.number_input(
                        "어린이",
                        min_value=0,
                        value=0
                    )

                with col3:
                    babies = st.number_input(
                        "유아",
                        min_value=0,
                        value=0
                    )

                total_guests = adults + children + babies
                has_children = int((children + babies) > 0)
            
            with st.container(border=True):
                st.subheader("기타 예약 정보")

                option_col1, option_col2, option_col3, option_col4 = st.columns(4)

                with option_col1:
                    is_repeated_guest = int(st.checkbox("재방문 고객"))

                with option_col2:
                    has_agent = int(st.checkbox(
                        "여행사/에이전트 예약",
                        value=True
                    ))

                with option_col3:
                    has_company = int(st.checkbox("회사 예약"))

                with option_col4:
                    has_waiting_list = int(st.checkbox("대기 목록 존재"))

            submitted = st.form_submit_button(
                "예약 취소 위험 분석",
                type="primary",
                use_container_width=True
            )
            
            if submitted:

                input_data = pd.DataFrame([{
                    "hotel": hotel,
                    "lead_time": lead_time,
                    "arrival_date_month": arrival_date_month,
                    "meal": meal,
                    "market_segment": market_segment,
                    "is_repeated_guest": is_repeated_guest,
                    "previous_cancellations": previous_cancellations,
                    "previous_bookings_not_canceled": previous_bookings_not_canceled,
                    "reserved_room_type": reserved_room_type,
                    "customer_type": customer_type,
                    "adr": adr,
                    "required_car_parking_spaces": required_car_parking_spaces,
                    "total_of_special_requests": total_of_special_requests,
                    "total_guests": total_guests,
                    "has_children": has_children,
                    "has_agent": has_agent,
                    "has_company": has_company,
                    "total_nights": total_nights,
                    "has_waiting_list": has_waiting_list,
                    "country_group": country_group
                }])
                
                
                processed_input = preprocessor.transform(
                    input_data
                ).astype("float32")

                cancel_probability = float(
                    model.predict(
                        processed_input,
                        verbose=0
                    ).ravel()[0]
                )

                prediction = int(
                    cancel_probability
                    >= OPTIMIZED_THRESHOLD
                )

                booking_amount = adr * total_nights

                expected_loss = (
                    cancel_probability * booking_amount
                )

                if not np.isfinite(cancel_probability) or not 0 <= cancel_probability <= 1:
                    st.error("모델 확률이 유효하지 않습니다. 입력과 모델 파일을 확인하세요.")
                    st.stop()

                probability_text = (
                    "< 0.01%"
                    if cancel_probability < 0.0001
                    else (
                        f"{cancel_probability:.2%}"
                        if cancel_probability < 0.001
                        else f"{cancel_probability:.1%}"
                    )
                )

                expected_loss_text = (
                    "< 0.01"
                    if 0 < expected_loss < 0.01
                    else f"{expected_loss:,.2f}"
                )

                risk_level, risk_score = None, None
                if risk_thresholds is not None:
                    q50, q75 = risk_thresholds["q50"], risk_thresholds["q75"]
                    if expected_loss < q50:
                        risk_score = (expected_loss / q50) * 50
                        risk_level = "LOW"
                    elif expected_loss < q75:
                        risk_score = 50 + ((expected_loss - q50) / (q75 - q50)) * 25
                        risk_level = "MEDIUM"
                    else:
                        risk_score = 75 + ((expected_loss - q75) / q75) * 25
                        risk_level = "HIGH"
                    risk_score = min(max(risk_score, 0.0), 100.0)
                    
                    
        with result_col:
            with st.container(border=True):
                st.subheader("예측 결과")

                if submitted:
                    st.metric(
                        label="예약 취소 확률",
                        value=probability_text
                    )

                    st.progress(
                        min(cancel_probability, 1.0)
                    )
                    
                    st.divider()

                    score_col1, score_col2 = st.columns(2)
                    if risk_error:
                        st.warning(risk_error)

                    with score_col1:
                        st.markdown(
                            '<div class="centered-result-label">'
                            '관리 우선순위 점수'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        score_text = (
                            f"{risk_score:.0f} / 100"
                            if risk_score is not None
                            else "—"
                        )

                        st.markdown(
                            f'<div class="priority-result-value">'
                            f'{score_text}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    with score_col2:
                        st.markdown(
                            '<div class="centered-result-label">'
                            '손실 위험 등급'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        if risk_level == "LOW":
                            st.markdown(
                                '<div class="risk-badge low centered">LOW</div>',
                                unsafe_allow_html=True
                            )

                        elif risk_level == "MEDIUM":
                            st.markdown(
                                '<div class="risk-badge medium centered">MEDIUM</div>',
                                unsafe_allow_html=True
                            )

                        elif risk_level == "HIGH":
                            st.markdown(
                                '<div class="risk-badge high centered">HIGH</div>',
                                unsafe_allow_html=True
                            )

                        else:
                            st.write("—")
                    if risk_score is not None:
                        with st.container(key="priority_progress"):
                            st.progress(min(risk_score / 100, 1.0))
                    
                    st.divider()

                    amount_col1, amount_col2 = st.columns(2)

                    with amount_col1:
                        st.markdown(
                            '<div class="centered-result-label">'
                            '예약 금액 지표'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f'<div class="amount-result-value">'
                            f'{booking_amount:,.2f}'
                            f'</div>',
                            unsafe_allow_html=True
                        )

                    with amount_col2:
                        st.markdown(
                            '<div class="centered-result-label">'
                            '예상 손실 지표'
                            '</div>',
                            unsafe_allow_html=True
                        )

                        st.markdown(
                            f'<div class="amount-result-value">'
                            f'{expected_loss_text}'
                            f'</div>',
                            unsafe_allow_html=True
                        )
                        
                else:
                    with st.container(border=True):
                        st.info(
                            "왼쪽의 예약 정보를 입력한 후 "
                            "'취소 위험 분석' 버튼을 눌러주세요."
                        )

# 아래 대시보드는 CSV/PNG를 읽기만 합니다. 재학습·파일 덮어쓰기를 하지 않습니다.
METRICS = ["accuracy", "precision", "recall", "f1", "roc_auc"]
LABELS = ["Accuracy", "Precision", "Recall", "F1", "ROC-AUC"]
COLORS = ["#4C78A8", "#F58518"]


@st.cache_data
def read_csv_cached(path, modified_ns):
    # 수정 시각을 캐시 키에 포함하여 파일 교체 후 오래된 결과 표시 방지
    return pd.read_csv(path)


def dashboard_csv(relative_path, required=()):
    path = ROOT / relative_path
    if not path.is_file():
        st.warning(f"자료 없음: {relative_path} — 이 영역만 표시하지 않습니다.")
        return None
    try:
        frame = read_csv_cached(str(path), path.stat().st_mtime_ns)
        missing = set(required) - set(frame.columns)
        if missing:
            raise ValueError(f"필수 열 없음: {sorted(missing)}")
        if frame.empty:
            raise ValueError("빈 CSV입니다.")
        return frame
    except Exception as error:
        st.warning(f"자료 확인 필요: {relative_path} | {error}")
        return None


def metric_rows(frame, threshold=0.5):
    return frame.loc[np.isclose(frame["threshold"], threshold)].copy()


def show_table(frame, name):
    st.dataframe(frame.round(4), hide_index=True, use_container_width=True)
    st.download_button(
        "이 표 CSV 다운로드", frame.to_csv(index=False).encode("utf-8-sig"),
        file_name=name, mime="text/csv", key=name
    )


def show_plot(fig):
    fig.tight_layout()
    st.pyplot(fig)
    plt.close(fig)


def performance_plot(frame, title):
    fig, ax = plt.subplots(figsize=(12, 4.5))
    x = np.arange(len(METRICS))
    width = 0.8 / len(frame)
    for i, (_, row) in enumerate(frame.iterrows()):
        bars = ax.bar(
            x + (i - (len(frame) - 1) / 2) * width,
            [float(row[m]) for m in METRICS], width,
            label=str(row["model"]), color=COLORS[i % len(COLORS)]
        )
        ax.bar_label(bars, fmt="%.4f", padding=3, fontsize=9)
    ax.set_xticks(x, LABELS)
    ax.set_ylim(0, 1.05)
    ax.set_ylabel("Score")
    ax.set_title(title)
    ax.legend(loc="lower right")
    ax.grid(axis="y", alpha=0.2)
    ax.set_axisbelow(True)
    show_plot(fig)


def saved_image(relative_path, caption):
    path = ROOT / relative_path
    if path.is_file():
        st.image(str(path), caption=caption, use_container_width=True)
    else:
        st.warning(f"그림 파일 없음: {relative_path}")


with dashboard_tab:
    st.subheader("MLP 모델 인사이트 대시보드")
    st.caption("모델 선정: Validation | 최종 비교: Common Test")

    common = dashboard_csv(
        "model_dl/comparison/verified_test/rf_mlp_common_test_metrics.csv",
        ["model", "data_split", "n_samples", "threshold", *METRICS, "tn", "fp", "fn", "tp"]
    )
    if common is not None:
        common = metric_rows(common)
        valid_common = (
            len(common) == 2
            and set(common["model"]) == {"Random Forest", "MLP"}
            and common["n_samples"].eq(23712).all()
            and common["data_split"].eq("Common Test").all()
            and common[["tn", "fp", "fn", "tp"]].sum(axis=1).eq(23712).all()
        )
        if not valid_common:
            st.error("공통 Test 비교표의 모델명·분할·표본 수·혼동행렬 합계를 확인하세요. 옛 RF 결과로 대체하지 않습니다.")
            common = None
        else:
            common = common.set_index("model").loc[["Random Forest", "MLP"]].reset_index()
            mlp_result = common.loc[common["model"].eq("MLP")].iloc[0]
            for column, metric, label in zip(st.columns(5), METRICS, LABELS):
                with column:
                    st.metric(label, f"{mlp_result[metric]:.4f}")
            st.caption("최종 MLP · 공통 Test 23,712건 · 임계값 0.5")

    basic_tab, search_tab, test_tab, threshold_tab, reference_tab = st.tabs([
        "기본 → 개선 MLP", "탐색·모델 구조", "RF vs MLP · Test",
        "임계값 조정", "손실 위험등급"
    ])

    with basic_tab:
        st.markdown("### 기본 MLP와 개선 MLP의 Validation 성능")
        baseline = dashboard_csv(
            "model_dl/baseline_recovery/mlp_baseline_validation_metrics.csv",
            ["threshold", *METRICS]
        )
        improved = dashboard_csv(
            "model_dl/validation/mlp_validation_threshold_05.csv",
            ["threshold", *METRICS]
        )
        if baseline is not None and improved is not None:
            b, m = metric_rows(baseline), metric_rows(improved)
            if len(b) == len(m) == 1:
                comparison = pd.DataFrame([
                    {"model": label, "data_split": "Validation", "threshold": 0.5,
                     **{key: float(row[key]) for key in METRICS}}
                    for label, row in [("Baseline MLP", b.iloc[0]), ("Improved MLP", m.iloc[0])]
                ])
                performance_plot(comparison, "Baseline vs Improved — Validation (0.5)")
                show_table(comparison, "baseline_vs_improved_validation.csv")
                st.caption("기본: 최고 체크포인트 | 개선: 107 Epoch | 단일 학습 Validation 평가")
            else:
                st.warning("각 평가 CSV에는 임계값 0.5 결과가 한 행씩 있어야 합니다.")

        st.markdown("### 학습 곡선")
        histories = []
        for label, path, color in [
            ("Baseline", "model_dl/baseline_recovery/mlp_baseline_log.csv", COLORS[0]),
            ("Improved", "model_dl/validation/mlp_validation_history.csv", COLORS[1])
        ]:
            history = dashboard_csv(path, ["loss", "val_loss", "accuracy", "val_accuracy", "roc_auc", "val_roc_auc"])
            if history is not None:
                histories.append((label, history, color))
        if histories:
            fig, axes = plt.subplots(1, 3, figsize=(18, 5))
            for ax, metric, title in zip(axes, ["loss", "accuracy", "roc_auc"], ["Loss", "Accuracy", "ROC-AUC"]):
                for label, history, color in histories:
                    epochs = np.arange(1, len(history) + 1)
                    ax.plot(epochs, history[metric], "--", color=color, alpha=0.65, label=f"{label} Train")
                    ax.plot(epochs, history[f"val_{metric}"], color=color, label=f"{label} Validation")
                ax.set(title=title, xlabel="Epoch", ylabel=title)
                ax.legend(fontsize=8)
                ax.grid(alpha=0.2)
            show_plot(fig)
            summaries = []
            for label, history, _ in histories:
                summaries.append({
                    "model": label, "epochs_run": len(history),
                    "min_val_loss_epoch": int(np.argmin(history["val_loss"])) + 1,
                    "max_val_accuracy_epoch": int(np.argmax(history["val_accuracy"])) + 1,
                    "max_val_roc_auc_epoch": int(np.argmax(history["val_roc_auc"])) + 1
                })
            st.dataframe(pd.DataFrame(summaries), hide_index=True, use_container_width=True)
        st.caption("Loss 구성 — 기본: BCE | 개선: BCE + L2")

    with search_tab:
        st.markdown("### 모델 개선 과정과 반복 실험")
        search_frames = []
        for label, path in [
            ("광범위 탐색", "model_dl/search/mlp_search_results.csv"),
            ("집중 탐색", "model_dl/focused/focused_search_results.csv"),
            ("반복 검증", "model_dl/stability/stability_results.csv")
        ]:
            frame = dashboard_csv(path, ["status"])
            if frame is not None:
                ok = frame.loc[frame["status"].eq("ok")]
                search_frames.append({"실험": label, "정상 완료": len(ok), "오류 기록": int(frame["status"].eq("error").sum())})
        if search_frames:
            st.dataframe(pd.DataFrame(search_frames), hide_index=True, use_container_width=True)
        st.caption("탐색 변수: 층 수·너비 / Dropout / L2 / 학습률 / 배치 크기 / BatchNorm / 옵티마이저")
        summary = dashboard_csv("model_dl/stability/stability_summary.csv", ["units", "optimizer", "runs", "mean_roc_auc", "std_roc_auc"])
        if summary is not None:
            summary = summary.sort_values("mean_roc_auc", ascending=False).reset_index(drop=True)
            fig, ax = plt.subplots(figsize=(11, 5))
            labels = [f"#{i+1} {row['units']} / {row['optimizer']}" for i, row in summary.iterrows()]
            ax.errorbar(summary["mean_roc_auc"], np.arange(len(summary)), xerr=summary["std_roc_auc"], fmt="o", capsize=4, color=COLORS[0])
            ax.set_yticks(np.arange(len(summary)), labels)
            ax.invert_yaxis()
            ax.set(xlabel="Validation ROC-AUC (mean ± seed standard deviation)", title="Repeated Runs — Same Validation Split")
            ax.grid(axis="x", alpha=0.2)
            show_plot(fig)
            st.caption("동일 Validation 분할 · 시드별 평균 ± 표준편차 · 가로축 확대")
            show_table(summary, "mlp_stability_summary.csv")
        with st.expander("전체 후보 결과 보기"):
            all_results = dashboard_csv("model_dl/combined/all_unique_results.csv", ["val_roc_auc"])
            if all_results is not None:
                st.write(f"중복 제거 후 후보 수: {len(all_results):,}")
                show_table(all_results.sort_values("val_roc_auc", ascending=False), "mlp_all_unique_results.csv")
        st.markdown("### 최종 MLP 구조·설정")
        st.code("Input 64 → Dense 256 → BN → Dropout\n         → Dense 128 → BN → Dropout\n         → Dense  64 → BN → Dropout\n         → Dense   1 (Sigmoid)", language="text")
        st.caption("ReLU · Dropout 0.4 · L2 0.0005 · Adam 0.0005 · 배치 512 · 전체 Train 107 Epoch")


    with test_tab:
        st.markdown("### 검증 완료한 공통 Test 비교")
        st.caption("23,712건 · 임계값 0.5 · 예약 인덱스·정답·재예측 대조 완료")
        if common is not None:
            performance_plot(common, "Random Forest vs MLP — Common Test (0.5)")
            show_table(common, "rf_mlp_common_test_metrics.csv")
            rf_row = common.loc[common["model"].eq("Random Forest")].iloc[0]
            mlp_row = common.loc[common["model"].eq("MLP")].iloc[0]
            st.markdown("#### 성능 차이 · MLP − Random Forest")
            delta = pd.DataFrame({
                "지표": LABELS, "Random Forest": [rf_row[m] for m in METRICS],
                "MLP": [mlp_row[m] for m in METRICS],
                "차이 (MLP − RF)": [mlp_row[m]-rf_row[m] for m in METRICS]
            })
            show_table(delta, "rf_mlp_metric_differences.csv")
            diff_tp, diff_fp = st.columns(2)
            diff_tp.metric("취소 예약 탐지 건수 차이 (TP)", f"{int(mlp_row['tp'] - rf_row['tp']):+,}건")
            diff_fp.metric("정상 예약 오탐 건수 차이 (FP)", f"{int(mlp_row['fp'] - rf_row['fp']):+,}건")
            fig, ax = plt.subplots(figsize=(10, 3.5))
            changes = [(mlp_row[m]-rf_row[m])*100 for m in METRICS[:4]]
            bars = ax.barh(LABELS[:4], changes, color=[COLORS[1] if value >= 0 else COLORS[0] for value in changes])
            ax.bar_label(bars, fmt="%+.2f", padding=4)
            ax.axvline(0, color="gray", linewidth=1)
            ax.set(xlabel="MLP − Random Forest (percentage points)", title="Common Test — Metric Difference")
            ax.margins(x=.25)
            show_plot(fig)
            st.dataframe(pd.DataFrame({
                "항목": ["모델 계열", "학습 방식", "확률 산출"],
                "Random Forest": ["결정트리 앙상블", "여러 트리의 분할 학습", "트리별 클래스 확률 평균"],
                "MLP": ["다층 신경망", "역전파·Adam", "출력층 Sigmoid"]
            }), hide_index=True, use_container_width=True)
            saved_image("model_dl/comparison/verified_test/rf_mlp_common_test_roc.png", "공통 Test ROC 곡선 — 분류 임계값과 무관한 확률 순위 비교")
            saved_image("model_dl/comparison/verified_test/rf_mlp_common_test_confusion.png", "공통 Test 혼동행렬 — 두 모델 모두 임계값 0.5")

    with threshold_tab:
        st.markdown("### Validation에서 Accuracy 기준으로 임계값 선정")
        threshold_search = dashboard_csv("model_dl/validation/mlp_threshold_search.csv", ["threshold", "accuracy", "precision", "recall", "f1"])
        if threshold_search is not None:
            threshold_search = threshold_search.sort_values("threshold")
            fig, ax = plt.subplots(figsize=(12, 4.5))
            for metric in METRICS[:-1]:
                ax.plot(threshold_search["threshold"], threshold_search[metric], label=metric.title())
            ax.axvline(0.5, color="gray", linestyle="--", label="Default 0.500")
            ax.axvline(OPTIMIZED_THRESHOLD, color="red", linestyle="--", label=f"Selected {OPTIMIZED_THRESHOLD:.3f}")
            ax.set(xlabel="Threshold", ylabel="Validation Score", ylim=(0, 1.02))
            ax.legend(ncol=3)
            ax.grid(alpha=0.2)
            show_plot(fig)
        threshold_comparison = dashboard_csv("model_dl/validation/mlp_threshold_comparison.csv", ["threshold", *METRICS])
        if threshold_comparison is not None:
            show_table(threshold_comparison, "mlp_validation_threshold_comparison.csv")
        st.markdown("### 확정 임계값의 최종 Test 결과")
        test_metrics = dashboard_csv("model_dl/final/final_mlp_test_metrics.csv", ["threshold", *METRICS])
        if test_metrics is not None:
            show_table(test_metrics, "final_mlp_test_metrics.csv")
            saved_image("model_dl/final/final_mlp_confusion_matrices.png", "최종 MLP Test — 기본 임계값과 Validation에서 선정한 임계값")
        st.caption("선정 기준: Validation Accuracy | 평가: Test | 모델 예측확률 고정")

    with reference_tab:
        st.markdown("### MLP 손실 위험등급")
        st.caption("예상 손실 지표 = 취소확률 × ADR × 숙박일 | 금액 단위: 원본 ADR 기준")
        if risk_error:
            st.warning(risk_error)
        else:
            q50, q75 = risk_thresholds["q50"], risk_thresholds["q75"]
            st.dataframe(pd.DataFrame({
                "등급": ["LOW", "MEDIUM", "HIGH"],
                "예상 손실 지표 범위": [f"{q50:,.6f} 미만", f"{q50:,.6f} 이상 ~ {q75:,.6f} 미만", f"{q75:,.6f} 이상"],
                "Validation 기준 구간": ["하위 50%", "상위 25~50%", "상위 25%"]
            }), hide_index=True, use_container_width=True)
            st.caption("경계 선정: Validation | Test 평가: 경계 고정")
            risk_frames = []
            for name, filename in [("Validation", "mlp_risk_validation_summary.csv"), ("Common Test", "mlp_risk_test_summary.csv")]:
                frame = dashboard_csv("model_dl/risk/" + filename, [
                    "risk_level", "n_samples", "actual_cancel_rate", "mean_expected_loss",
                    "booking_share", "canceled_amount_capture_rate", "cancellation_capture_rate"])
                if frame is not None:
                    risk_frames.append((name, frame.set_index("risk_level").loc[["LOW", "MEDIUM", "HIGH"]]))
            if risk_frames:
                fig, axes = plt.subplots(1, 2, figsize=(13, 4.5))
                for axis, field, title in zip(axes, ["actual_cancel_rate", "canceled_amount_capture_rate"],
                                            ["Actual Cancellation Rate", "Share of Canceled Booking Amount"]):
                    x = np.arange(3)
                    width = .8 / len(risk_frames)
                    for i, (name, frame) in enumerate(risk_frames):
                        bars = axis.bar(x + (i-(len(risk_frames)-1)/2)*width, frame[field]*100,
                                        width, label=name, color=COLORS[i])
                        axis.bar_label(bars, fmt="%.1f%%", padding=3, fontsize=9)
                    axis.set_xticks(x, ["LOW", "MEDIUM", "HIGH"])
                    axis.set(title=title, ylabel="%", ylim=(0,110))
                    axis.legend()
                show_plot(fig)
                rows = []
                for name, frame in risk_frames:
                    for label, levels in [("HIGH", ["HIGH"]), ("MEDIUM + HIGH", ["MEDIUM", "HIGH"])]:
                        part = frame.loc[levels]
                        rows.append({"데이터": name, "관리 대상": label,
                                     "관리 건수": int(part["n_samples"].sum()),
                                     "관리 비율(%)": part["booking_share"].sum()*100,
                                     "취소 건수 포착률(%)": part["cancellation_capture_rate"].sum()*100,
                                     "취소 예약금액 포착률(%)": part["canceled_amount_capture_rate"].sum()*100})
                st.markdown("#### 관리 범위별 누적 포착률")
                show_table(pd.DataFrame(rows), "mlp_risk_capture_summary.csv")
                with st.expander("등급별 상세 집계"):
                    for name, frame in risk_frames:
                        st.markdown(f"**{name}**")
                        show_table(frame.reset_index(), f"mlp_risk_{name.replace(' ', '_').lower()}.csv")
            st.markdown("#### Validation 관리 비율별 포착률")
            saved_image("model_dl/comparison/capture_analysis/mlp_validation_capture.png", "Validation · 예상 손실 내림차순")
        st.caption("지표 정의: 취소 예약금액 = 실제 취소 여부 × ADR × 숙박일")
        with st.expander("기존 RF 참고 분석 — MLP 중요도·공통 Test 결과가 아님"):
            st.caption("이전 RF 분석 기록")
            st.dataframe(rf_results.round(4), hide_index=True, use_container_width=True)
            top = importance_df.sort_values("Importance")
            fig, ax = plt.subplots(figsize=(9, 5))
            ax.barh(top["Feature"], top["Importance"], color=COLORS[0])
            ax.set(xlabel="RF Permutation Importance (historical reference)", title="RF Reference — Not MLP Feature Importance")
            show_plot(fig)
            st.dataframe(pd.DataFrame({
                "기존 RF 위험등급": ["LOW", "MEDIUM", "HIGH"],
                "기존 RF Validation 실제 취소율(%)": [10.3, 53.8, 74.5],
                "기존 RF 평균 예상 손실 지표": [22.269, 124.273, 387.696]
            }), hide_index=True, use_container_width=True)
