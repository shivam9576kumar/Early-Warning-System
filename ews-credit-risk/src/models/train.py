"""Model training, hyperparameter configuration, and probability calibration for corporate credit risk.

Models trained:
1. Baseline: Logistic Regression with StandardScaler, class_weight='balanced', L2 penalty.
2. Primary: LightGBM Classifier with class-imbalance weight (scale_pos_weight), max_depth 4-6.
3. Secondary: XGBoost Classifier with scale_pos_weight.
4. Probability Calibration: Isotonic Regression on held-out validation fold.
"""

from typing import Any, Dict, Optional, Tuple
import lightgbm as lgb
import numpy as np
import pandas as pd
from sklearn.isotonic import IsotonicRegression
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
import xgboost as xgb

# Model class aliases
XGBClassifier = xgb.XGBClassifier
LGBMClassifier = lgb.LGBMClassifier


class PrefitIsotonicCalibrator:
    """Isotonic regression probability calibrator for prefit binary classifiers.

    Calibrates raw model scores/probabilities using a held-out calibration fold
    without altering base estimator weights or triggering cross-validation refitting.
    """

    def __init__(self, base_estimator: Any):
        self.base_estimator = base_estimator
        self.calibrator = IsotonicRegression(out_of_bounds="clip", y_min=0.0, y_max=1.0)

    def fit(self, X_val: np.ndarray, y_val: np.ndarray) -> "PrefitIsotonicCalibrator":
        raw_prob = self.base_estimator.predict_proba(X_val)[:, 1]
        self.calibrator.fit(raw_prob, y_val)
        return self

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        raw_prob = self.base_estimator.predict_proba(X)[:, 1]
        cal_prob = self.calibrator.predict(raw_prob)
        # Ensure 2D probability array [P(y=0), P(y=1)]
        cal_prob = np.clip(cal_prob, 0.0, 1.0)
        return np.vstack([1.0 - cal_prob, cal_prob]).T

    def predict(self, X: np.ndarray, threshold: float = 0.5) -> np.ndarray:
        prob_1 = self.predict_proba(X)[:, 1]
        return (prob_1 >= threshold).astype(int)


def train_logistic_baseline(
    X_train: np.ndarray,
    y_train: np.ndarray,
    C: float = 0.1,
    random_state: int = 42,
) -> Pipeline:
    """Trains a regularized Logistic Regression baseline with StandardScaler."""
    pipe = Pipeline([
        ("scaler", StandardScaler()),
        (
            "clf",
            LogisticRegression(
                C=C,
                class_weight="balanced",
                solver="lbfgs",
                max_iter=1000,
                random_state=random_state,
            ),
        ),
    ])
    pipe.fit(X_train, y_train)
    return pipe


def train_lightgbm_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    max_depth: int = 5,
    learning_rate: float = 0.03,
    n_estimators: int = 350,
    random_state: int = 42,
) -> lgb.LGBMClassifier:
    """Trains LightGBM with imbalance compensation (scale_pos_weight) and early stopping."""
    neg_count = np.sum(y_train == 0)
    pos_count = max(np.sum(y_train == 1), 1)
    scale_pos_weight = float(neg_count) / float(pos_count)

    model = lgb.LGBMClassifier(
        objective="binary",
        max_depth=max_depth,
        num_leaves=2**max_depth - 1,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        verbose=-1,
    )

    if X_val is not None and y_val is not None:
        model.fit(
            X_train,
            y_train,
            eval_set=[(X_val, y_val)],
            callbacks=[lgb.early_stopping(stopping_rounds=30, verbose=False)],
        )
    else:
        model.fit(X_train, y_train)

    return model


def train_xgboost_model(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: Optional[np.ndarray] = None,
    y_val: Optional[np.ndarray] = None,
    max_depth: int = 5,
    learning_rate: float = 0.03,
    n_estimators: int = 350,
    random_state: int = 42,
) -> xgb.XGBClassifier:
    """Trains XGBoost comparison model with scale_pos_weight."""
    neg_count = np.sum(y_train == 0)
    pos_count = max(np.sum(y_train == 1), 1)
    scale_pos_weight = float(neg_count) / float(pos_count)

    model = xgb.XGBClassifier(
        objective="binary:logistic",
        max_depth=max_depth,
        learning_rate=learning_rate,
        n_estimators=n_estimators,
        scale_pos_weight=scale_pos_weight,
        subsample=0.8,
        colsample_bytree=0.8,
        random_state=random_state,
        n_jobs=-1,
        eval_metric="logloss",
        early_stopping_rounds=30 if (X_val is not None and y_val is not None) else None,
    )

    if X_val is not None and y_val is not None:
        model.fit(X_train, y_train, eval_set=[(X_val, y_val)], verbose=False)
    else:
        model.fit(X_train, y_train, verbose=False)

    return model


def calibrate_model_isotonic(
    base_model: Any,
    X_val: np.ndarray,
    y_val: np.ndarray,
) -> PrefitIsotonicCalibrator:
    """Calibrates predicted probabilities using Isotonic Regression on held-out validation fold."""
    calibrator = PrefitIsotonicCalibrator(base_model)
    calibrator.fit(X_val, y_val)
    return calibrator


class EnsembleClassifier:
    """High-accuracy weighted ensemble combining XGBoost and LightGBM."""

    def __init__(self, xgb_model: Any, lgb_model: Any, w_xgb: float = 0.60, w_lgb: float = 0.40):
        self.xgb_model = xgb_model
        self.lgb_model = lgb_model
        self.w_xgb = w_xgb
        self.w_lgb = w_lgb
        self.base_estimator = xgb_model

    def predict_proba(self, X: np.ndarray) -> np.ndarray:
        p_xgb = self.xgb_model.predict_proba(X)[:, 1]
        p_lgb = self.lgb_model.predict_proba(X)[:, 1]
        p_ens = self.w_xgb * p_xgb + self.w_lgb * p_lgb
        p_ens = np.clip(p_ens, 0.0, 1.0)
        return np.vstack([1.0 - p_ens, p_ens]).T

    def predict(self, X: np.ndarray, threshold: float = 0.50) -> np.ndarray:
        return (self.predict_proba(X)[:, 1] >= threshold).astype(int)
