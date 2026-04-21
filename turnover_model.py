"""
Employee Turnover Prediction Model
Inputs : 年齢, 性別, 勤務年数, 夜勤回数, ストレス指標
Output : 離職確率 (0.0 ~ 1.0)
"""

import numpy as np
import pandas as pd
from sklearn.ensemble import GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import Pipeline
from sklearn.model_selection import train_test_split
from sklearn.metrics import roc_auc_score, classification_report
import joblib
import os

MODEL_PATH = os.path.join(os.path.dirname(__file__), "turnover_model.pkl")


def generate_training_data(n_samples: int = 5000, seed: int = 42) -> pd.DataFrame:
    rng = np.random.default_rng(seed)

    age = rng.integers(22, 66, n_samples).astype(float)
    gender = rng.integers(0, 2, n_samples).astype(float)          # 0=男性, 1=女性
    tenure = rng.integers(0, 31, n_samples).astype(float)          # 勤務年数
    night_shifts = rng.integers(0, 21, n_samples).astype(float)    # 月あたり夜勤回数
    stress = rng.uniform(1.0, 10.0, n_samples)                     # ストレス指標

    # 離職確率の論理的な計算
    logit = (
        -3.5
        + 0.03 * np.maximum(age - 45, 0)    # 45歳超で年齢が上がるにつれリスク増
        - 0.15 * tenure                      # 勤務年数が長いほどリスク減
        + 0.12 * night_shifts               # 夜勤回数が多いほどリスク増
        + 0.35 * stress                     # ストレスが高いほどリスク増
        + rng.normal(0, 0.5, n_samples)     # ノイズ
    )
    prob = 1 / (1 + np.exp(-logit))
    label = (rng.uniform(0, 1, n_samples) < prob).astype(int)

    return pd.DataFrame({
        "年齢": age,
        "性別": gender,
        "勤務年数": tenure,
        "夜勤回数": night_shifts,
        "ストレス指標": stress,
        "離職": label,
    })


def train(save: bool = True) -> Pipeline:
    df = generate_training_data()
    X = df.drop(columns=["離職"])
    y = df["離職"]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )

    model = Pipeline([
        ("scaler", StandardScaler()),
        ("clf", GradientBoostingClassifier(
            n_estimators=200,
            max_depth=4,
            learning_rate=0.05,
            random_state=42,
        )),
    ])
    model.fit(X_train, y_train)

    proba = model.predict_proba(X_test)[:, 1]
    auc = roc_auc_score(y_test, proba)
    print(f"AUC-ROC: {auc:.4f}")
    print(classification_report(y_test, model.predict(X_test)))

    if save:
        joblib.dump(model, MODEL_PATH)
        print(f"モデルを保存しました: {MODEL_PATH}")

    return model


def load_model() -> Pipeline:
    if not os.path.exists(MODEL_PATH):
        raise FileNotFoundError(
            f"モデルファイルが見つかりません: {MODEL_PATH}\n"
            "先に `python train.py` を実行してください。"
        )
    return joblib.load(MODEL_PATH)


def predict(
    age: float,
    gender: int,
    tenure: float,
    night_shifts: float,
    stress: float,
    model: Pipeline | None = None,
) -> float:
    """離職確率 (0.0 ~ 1.0) を返す。"""
    if model is None:
        model = load_model()

    X = pd.DataFrame([{
        "年齢": age,
        "性別": gender,
        "勤務年数": tenure,
        "夜勤回数": night_shifts,
        "ストレス指標": stress,
    }])
    return float(model.predict_proba(X)[0, 1])
