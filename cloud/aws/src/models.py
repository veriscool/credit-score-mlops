"""Build candidate models for the credit-score classifier.

A factory returning named candidate estimators, extracted from the project's
local ../../pipeline.py. Each model carries its own Optuna search space and
early-stopping logic because that's how the local champion (see
../../artifacts/best_model.json) was actually tuned -- keeping that logic
intact here is what lets build_models() reproduce the same macro-F1 the
local pipeline got.
"""

from __future__ import annotations

import logging
import os
from abc import ABC, abstractmethod
from dataclasses import dataclass
from typing import Any, Dict, Optional

import numpy as np
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

logger = logging.getLogger("models")


@dataclass
class TrainResult:
    name: str
    model: Any
    params: Dict[str, Any]
    cv_score: Optional[float] = None


class BaseModel(ABC):
    name: str = "base"
    requires_scaling: bool = False
    supports_sample_weight: bool = False
    sample_weight_key: str = "sample_weight"
    supports_early_stopping: bool = False
    early_stopping_rounds: int = 50

    def __init__(self, random_state: int = 42) -> None:
        self.random_state = random_state
        self.n_jobs = -1
        self.study_ = None

    @abstractmethod
    def build(self, params: Dict[str, Any]) -> Any:
        """Construct the (unfitted) estimator for these params."""

    @abstractmethod
    def search_space(self, trial) -> Dict[str, Any]:
        """Optuna search space for this model."""

    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """Tuned defaults used when n_trials == 0 (no Optuna search)."""

    # shared fit helper (sample weighting + optional early stopping)
    def fit_one(self, params: Dict[str, Any], X_tr, y_tr, sample_weight=None, eval_set=None):
        p = dict(params)
        fit_kwargs: Dict[str, Any] = {}
        if eval_set is not None and self.supports_early_stopping:
            p = self._inject_early_stopping(p)
            fit_kwargs.update(self._eval_fit_kwargs(eval_set))
        model = self.build(p)
        if sample_weight is not None:
            fit_kwargs[self.sample_weight_key] = sample_weight
        model.fit(X_tr, y_tr, **fit_kwargs)
        return model

    # subclasses that early-stop override these two hooks
    def _inject_early_stopping(self, params: Dict[str, Any]) -> Dict[str, Any]:
        return params

    def _eval_fit_kwargs(self, eval_set) -> Dict[str, Any]:
        return {}

    # tuning is shared logic for every model
    def tune(self, X, y, groups, n_trials: int, smote=None, n_jobs: int = 1, storage: Optional[str] = None) -> Dict[str, Any]:
        if n_trials <= 0:
            logger.info("[%s] tuning skipped", self.name)
            return self.default_params()

        import optuna
        from sklearn.metrics import f1_score
        from sklearn.model_selection import StratifiedGroupKFold
        optuna.logging.set_verbosity(optuna.logging.WARNING)

        def objective(trial):
            params = self.search_space(trial)
            cv = StratifiedGroupKFold(n_splits=3, shuffle=True,
                                      random_state=self.random_state)
            scores = []
            for step, (tr, va) in enumerate(cv.split(X, y, groups=groups)):
                X_tr, y_tr = X.iloc[tr], y.iloc[tr]
                X_va, y_va = X.iloc[va], y.iloc[va]
                if smote is not None:
                    X_tr, y_tr = clone(smote).fit_resample(X_tr, y_tr)
                sw = (compute_sample_weight("balanced", y_tr)
                      if self.supports_sample_weight else None)
                eval_set = [(X_va, y_va)] if self.supports_early_stopping else None
                model = self.fit_one(params, X_tr, y_tr,
                                     sample_weight=sw, eval_set=eval_set)
                scores.append(f1_score(y_va, model.predict(X_va), average="macro"))
                trial.report(float(np.mean(scores)), step=step)
                if trial.should_prune():
                    raise optuna.TrialPruned()
            return float(np.mean(scores))

        sampler = optuna.samplers.TPESampler(seed=self.random_state)
        pruner = optuna.pruners.MedianPruner(n_startup_trials=5, n_warmup_steps=1)

        prev_n_jobs = self.n_jobs
        if n_jobs and n_jobs > 1:
            self.n_jobs = max(1, (os.cpu_count() or 1) // n_jobs)

        study = None
        if storage:
            try:
                study = optuna.create_study(
                    study_name=self.name, storage=storage, load_if_exists=True,
                    direction="maximize", sampler=sampler, pruner=pruner)
            except Exception as exc:
                logger.warning("[%s] persistent study unavailable (%s); using in-memory.", self.name, exc)
        if study is None:
            study = optuna.create_study(
                study_name=self.name, direction="maximize",
                sampler=sampler, pruner=pruner)

        if not study.trials:
            try:
                study.enqueue_trial(self.default_params())
            except Exception as exc:
                logger.warning("[%s] could not enqueue defaults: %s", self.name, exc)

        try:
            study.optimize(objective, n_trials=n_trials, n_jobs=n_jobs, show_progress_bar=True)
        finally:
            self.n_jobs = prev_n_jobs

        self.study_ = study
        n_pruned = len([t for t in study.trials if t.state.name == "PRUNED"])
        logger.info("[%s] best CV macro-F1=%.4f (%d trials, %d pruned)", self.name, study.best_value, len(study.trials), n_pruned)
        return {**self.default_params(), **study.best_params}

    def fit_final(self, X, y, params: Dict[str, Any], smote=None, groups=None) -> TrainResult:
        eval_set = None
        X_core, y_core = X, y
        if self.supports_early_stopping:
            X_core, y_core, X_val, y_val = self._early_stop_holdout(X, y, groups)
            eval_set = [(X_val, y_val)]

        X_fit, y_fit = X_core, y_core

        if smote is not None:
            X_fit, y_fit = clone(smote).fit_resample(X_core, y_core)

        sw = (compute_sample_weight("balanced", y_fit) if self.supports_sample_weight else None)

        model = self.fit_one(params, X_fit, y_fit, sample_weight=sw, eval_set=eval_set)

        return TrainResult(name=self.name, model=model, params=params)

    def _early_stop_holdout(self, X, y, groups):
        if groups is not None:
            from sklearn.model_selection import StratifiedGroupKFold
            sgkf = StratifiedGroupKFold(n_splits=6, shuffle=True, random_state=self.random_state)
            core_idx, val_idx = next(sgkf.split(X, y, groups=groups))
        else:
            from sklearn.model_selection import train_test_split
            core_idx, val_idx = train_test_split(np.arange(len(X)), test_size=0.15, stratify=y, random_state=self.random_state)
        return (X.iloc[core_idx], y.iloc[core_idx], X.iloc[val_idx], y.iloc[val_idx])


class XGBModel(BaseModel):
    name = "xgboost"
    supports_sample_weight = True
    supports_early_stopping = True

    def build(self, params):
        import xgboost as xgb
        return xgb.XGBClassifier(**{
            **params, "objective": "multi:softprob", "num_class": 3,
            "eval_metric": "mlogloss", "tree_method": "hist",
            "random_state": self.random_state, "n_jobs": self.n_jobs,
        })

    def _inject_early_stopping(self, params):
        return {**params, "early_stopping_rounds": self.early_stopping_rounds}

    def _eval_fit_kwargs(self, eval_set):
        return {"eval_set": eval_set, "verbose": False}

    def default_params(self):
        return {"n_estimators": 750, "max_depth": 6, "learning_rate": 0.004,
                "subsample": 0.9, "colsample_bytree": 0.8, "min_child_weight": 3}

    def search_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 2000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 10),
            "learning_rate": trial.suggest_float("learning_rate", 1e-4, 0.3, log=True),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "gamma": trial.suggest_float("gamma", 0.0, 5.0),
        }


class RandomForestModel(BaseModel):
    name = "random_forest"

    def build(self, params):
        from sklearn.ensemble import RandomForestClassifier
        return RandomForestClassifier(
            **params, random_state=self.random_state, n_jobs=self.n_jobs)

    def default_params(self):
        return {"n_estimators": 400, "max_depth": 25, "min_samples_split": 5,
                "min_samples_leaf": 2, "max_features": "sqrt",
                "class_weight": "balanced"}

    def search_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample", None]),
        }


class LightGBMModel(BaseModel):
    name = "lightgbm"
    supports_early_stopping = True

    def build(self, params):
        import lightgbm as lgb
        return lgb.LGBMClassifier(**{
            **params, "objective": "multiclass", "num_class": 3,
            "metric": "multi_logloss", "verbosity": -1,
            "random_state": self.random_state, "n_jobs": self.n_jobs,
        })

    def _eval_fit_kwargs(self, eval_set):
        import lightgbm as lgb
        return {"eval_set": eval_set,
                "callbacks": [lgb.early_stopping(self.early_stopping_rounds,
                                                 verbose=False),
                              lgb.log_evaluation(period=0)]}

    def default_params(self):
        return {"n_estimators": 500, "max_depth": 8, "learning_rate": 0.05,
                "num_leaves": 63, "subsample": 0.9, "colsample_bytree": 0.8,
                "min_child_samples": 20, "is_unbalance": True}

    def search_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 200, 1000, step=50),
            "max_depth": trial.suggest_int("max_depth", 3, 12),
            "learning_rate": trial.suggest_float("learning_rate", 1e-3, 0.3, log=True),
            "num_leaves": trial.suggest_int("num_leaves", 20, 300),
            "subsample": trial.suggest_float("subsample", 0.5, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.4, 1.0),
            "min_child_samples": trial.suggest_int("min_child_samples", 5, 100),
            "reg_alpha": trial.suggest_float("reg_alpha", 1e-8, 10.0, log=True),
            "reg_lambda": trial.suggest_float("reg_lambda", 1e-8, 10.0, log=True),
            "is_unbalance": trial.suggest_categorical("is_unbalance", [True, False]),
        }


class KNNModel(BaseModel):
    name = "knn"
    requires_scaling = True

    def build(self, params):
        from sklearn.neighbors import KNeighborsClassifier
        from sklearn.pipeline import Pipeline
        from sklearn.preprocessing import StandardScaler
        return Pipeline([
            ("scaler", StandardScaler()),
            ("knn", KNeighborsClassifier(**params, n_jobs=self.n_jobs)),
        ])

    def default_params(self):
        return {"n_neighbors": 15, "weights": "distance", "metric": "manhattan"}

    def search_space(self, trial):
        metric = trial.suggest_categorical("metric", ["euclidean", "manhattan", "minkowski"])
        params = {
            "n_neighbors": trial.suggest_int("n_neighbors", 3, 50),
            "weights": trial.suggest_categorical("weights", ["uniform", "distance"]),
            "metric": metric,
        }
        if metric == "minkowski":
            params["p"] = trial.suggest_int("p", 1, 4)
        return params


class ExtraTreesModel(BaseModel):
    name = "extra_trees"

    def build(self, params):
        from sklearn.ensemble import ExtraTreesClassifier
        return ExtraTreesClassifier(**params, random_state=self.random_state, n_jobs=self.n_jobs)

    def default_params(self):
        return {"n_estimators": 400, "max_depth": 25, "min_samples_split": 5,
                "min_samples_leaf": 2, "max_features": "sqrt",
                "class_weight": "balanced"}

    def search_space(self, trial):
        return {
            "n_estimators": trial.suggest_int("n_estimators", 100, 800, step=50),
            "max_depth": trial.suggest_int("max_depth", 5, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 20),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 20),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.3, 0.5]),
            "bootstrap": trial.suggest_categorical("bootstrap", [True, False]),
            "class_weight": trial.suggest_categorical(
                "class_weight", ["balanced", "balanced_subsample", None]),
        }


class DecisionTreeModel(BaseModel):
    name = "decision_tree"

    def build(self, params):
        from sklearn.tree import DecisionTreeClassifier
        return DecisionTreeClassifier(**params, random_state=self.random_state)

    def default_params(self):
        return {"max_depth": 15, "min_samples_split": 10,
                "min_samples_leaf": 5, "max_features": None,
                "criterion": "gini", "class_weight": "balanced"}

    def search_space(self, trial):
        return {
            "max_depth": trial.suggest_int("max_depth", 3, 40),
            "min_samples_split": trial.suggest_int("min_samples_split", 2, 30),
            "min_samples_leaf": trial.suggest_int("min_samples_leaf", 1, 30),
            "max_features": trial.suggest_categorical("max_features", ["sqrt", "log2", 0.5, None]),
            "criterion": trial.suggest_categorical("criterion", ["gini", "entropy"]),
            "class_weight": trial.suggest_categorical("class_weight", ["balanced", None]),
        }


def build_models(random_state: int = 42) -> Dict[str, BaseModel]:
    """Return a dict of {name: BaseModel instance} for all candidate models."""
    return {
        "xgboost": XGBModel(random_state),
        "random_forest": RandomForestModel(random_state),
        "lightgbm": LightGBMModel(random_state),
        "knn": KNNModel(random_state),
        "extra_trees": ExtraTreesModel(random_state),
        "decision_tree": DecisionTreeModel(random_state),
    }
