#!/usr/bin/env python3
"""Classical classifier zoo (sklearn + optional xgboost/lightgbm).

Every builder returns a probability-capable classifier (needed for ROC/PR-AUC and
for ranking candidates by class probability). Model selection happens in
:mod:`plm_classifier.search` via the size-filtered registry.
"""

from __future__ import annotations

from typing import Tuple

from sklearn.base import BaseEstimator
from sklearn.discriminant_analysis import LinearDiscriminantAnalysis, QuadraticDiscriminantAnalysis
from sklearn.ensemble import (
    ExtraTreesClassifier,
    HistGradientBoostingClassifier,
    RandomForestClassifier,
)
from sklearn.gaussian_process import GaussianProcessClassifier
from sklearn.gaussian_process.kernels import Matern
from sklearn.linear_model import LogisticRegression, SGDClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.neural_network import MLPClassifier
from sklearn.svm import SVC

try:
    from xgboost import XGBClassifier

    HAS_XGB = True
except Exception:  # pragma: no cover
    XGBClassifier = None
    HAS_XGB = False

try:
    from lightgbm import LGBMClassifier

    HAS_LGB = True
except Exception:  # pragma: no cover
    LGBMClassifier = None
    HAS_LGB = False


def build_classical_model(
    name: str,
    trial: "object",
    random_state: int,
    use_gpu: bool,
    n_features: int,
    n_samples_train: int,
) -> Tuple[str, BaseEstimator]:
    if name == "logreg":
        model = LogisticRegression(
            C=trial.suggest_float("lr_C", 1e-3, 1e3, log=True),
            class_weight=trial.suggest_categorical("lr_class_weight", [None, "balanced"]),
            solver="lbfgs",
            max_iter=3000,
            random_state=random_state,
        )
    elif name == "sgd":
        model = SGDClassifier(
            loss=trial.suggest_categorical("sgd_loss", ["log_loss", "modified_huber"]),
            penalty=trial.suggest_categorical("sgd_penalty", ["l2", "l1", "elasticnet"]),
            alpha=trial.suggest_float("sgd_alpha", 1e-7, 1e-1, log=True),
            class_weight=trial.suggest_categorical("sgd_class_weight", [None, "balanced"]),
            learning_rate="optimal",
            max_iter=5000,
            random_state=random_state,
        )
    elif name == "svc_rbf":
        model = SVC(
            kernel="rbf",
            C=trial.suggest_float("svc_C", 1e-2, 1e3, log=True),
            gamma=trial.suggest_categorical("svc_gamma", ["scale", "auto"]),
            class_weight=trial.suggest_categorical("svc_class_weight", [None, "balanced"]),
            probability=True,
            random_state=random_state,
        )
    elif name == "knn":
        model = KNeighborsClassifier(
            n_neighbors=trial.suggest_int("knn_n_neighbors", 3, 25),
            weights=trial.suggest_categorical("knn_weights", ["uniform", "distance"]),
            p=trial.suggest_int("knn_p", 1, 2),
        )
    elif name == "mlp":
        hidden = trial.suggest_categorical("mlp_hidden", ["64", "128", "256", "128_64", "256_128"])
        hidden_map = {"64": (64,), "128": (128,), "256": (256,), "128_64": (128, 64), "256_128": (256, 128)}
        model = MLPClassifier(
            hidden_layer_sizes=hidden_map[hidden],
            alpha=trial.suggest_float("mlp_alpha", 1e-6, 1e-1, log=True),
            learning_rate_init=trial.suggest_float("mlp_lr", 1e-4, 1e-2, log=True),
            max_iter=3000,
            early_stopping=True,
            random_state=random_state,
        )
    elif name == "rf":
        model = RandomForestClassifier(
            n_estimators=trial.suggest_int("rf_n_estimators", 100, 1000, step=100),
            max_depth=trial.suggest_int("rf_max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("rf_min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("rf_max_features", ["sqrt", "log2", None]),
            class_weight=trial.suggest_categorical("rf_class_weight", [None, "balanced", "balanced_subsample"]),
            n_jobs=1,
            random_state=random_state,
        )
    elif name == "extra_trees":
        model = ExtraTreesClassifier(
            n_estimators=trial.suggest_int("et_n_estimators", 100, 1000, step=100),
            max_depth=trial.suggest_int("et_max_depth", 3, 20),
            min_samples_leaf=trial.suggest_int("et_min_samples_leaf", 1, 10),
            max_features=trial.suggest_categorical("et_max_features", ["sqrt", "log2", None]),
            class_weight=trial.suggest_categorical("et_class_weight", [None, "balanced"]),
            n_jobs=1,
            random_state=random_state,
        )
    elif name == "hist_gb":
        model = HistGradientBoostingClassifier(
            learning_rate=trial.suggest_float("hgb_learning_rate", 1e-3, 0.3, log=True),
            max_depth=trial.suggest_int("hgb_max_depth", 2, 12),
            max_leaf_nodes=trial.suggest_int("hgb_max_leaf_nodes", 15, 255),
            l2_regularization=trial.suggest_float("hgb_l2", 1e-8, 1e1, log=True),
            min_samples_leaf=trial.suggest_int("hgb_min_samples_leaf", 5, 50),
            random_state=random_state,
        )
    elif name == "gpc":
        ls = trial.suggest_float("gpc_length_scale", 1e-1, 1e2, log=True)
        model = GaussianProcessClassifier(kernel=1.0 * Matern(length_scale=ls, nu=1.5), random_state=random_state)
    elif name == "gnb":
        model = GaussianNB(var_smoothing=trial.suggest_float("gnb_var_smoothing", 1e-12, 1e-3, log=True))
    elif name == "lda":
        model = LinearDiscriminantAnalysis(solver="lsqr", shrinkage=trial.suggest_float("lda_shrinkage", 0.0, 1.0))
    elif name == "qda":
        model = QuadraticDiscriminantAnalysis(reg_param=trial.suggest_float("qda_reg", 0.0, 1.0))
    elif name == "xgboost":
        if not HAS_XGB:
            raise ValueError("xgboost not installed")
        params = {
            "n_estimators": trial.suggest_int("xgb_n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("xgb_learning_rate", 1e-3, 0.3, log=True),
            "max_depth": trial.suggest_int("xgb_max_depth", 2, 10),
            "subsample": trial.suggest_float("xgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("xgb_colsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("xgb_reg_lambda", 1e-6, 10.0, log=True),
            "random_state": random_state,
            "n_jobs": 1,
            "tree_method": "hist",
            "eval_metric": "logloss",
        }
        if use_gpu:
            params["device"] = "cuda"
        model = XGBClassifier(**params)
    elif name == "lightgbm":
        if not HAS_LGB:
            raise ValueError("lightgbm not installed")
        params = {
            "n_estimators": trial.suggest_int("lgb_n_estimators", 100, 1000, step=100),
            "learning_rate": trial.suggest_float("lgb_learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("lgb_num_leaves", 15, 255),
            "subsample": trial.suggest_float("lgb_subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("lgb_colsample", 0.5, 1.0),
            "reg_lambda": trial.suggest_float("lgb_reg_lambda", 1e-6, 10.0, log=True),
            "class_weight": trial.suggest_categorical("lgb_class_weight", [None, "balanced"]),
            "random_state": random_state,
            "n_jobs": 1,
            "verbose": -1,
        }
        if use_gpu:
            params["device"] = "gpu"
        model = LGBMClassifier(**params)
    else:
        raise ValueError(f"Unknown classical classifier '{name}'")

    return name, model
