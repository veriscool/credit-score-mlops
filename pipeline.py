from __future__ import annotations

import argparse
import json
import logging
import os
import re
import warnings
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional, Tuple

import pickle
import numpy as np
import pandas as pd
from sklearn.base import clone
from sklearn.utils.class_weight import compute_sample_weight

warnings.filterwarnings("ignore")

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("pipeline")

# Label encoding
TARGET_COL = "Credit_Score"
ID_COL = "Customer_ID"
LABEL_NAMES = ["Poor", "Standard", "Good"]
CREDIT_SCORE_MAP = {"Poor": 0, "Standard": 1, "Good": 2}


# PREPROCESSING
class CreditDataPreprocessor:
    NUMERIC_PLACEHOLDER_COLS = [
        "Age", "Annual_Income", "Num_of_Loan", "Num_of_Delayed_Payment",
        "Changed_Credit_Limit", "Outstanding_Debt", "Amount_invested_monthly",
        "Monthly_Balance",
    ]

    GROUP_IMPUTE_COLS = [
        "Age", "Occupation", "Annual_Income", "Monthly_Inhand_Salary",
        "Num_of_Loan", "Type_of_Loan", "Credit_Mix", "Num_Credit_Inquiries",
        "Credit_History_Age", "Monthly_Balance",
    ]

    DROP_COLS = ["ID", "Name", "SSN"]

    CREDIT_MIX_MAP = {"Bad": 0, "Standard": 1, "Good": 2}
    SPEND_MAP = {"Low": 0, "High": 1}
    VALUE_MAP = {"Small": 0, "Medium": 1, "Large": 2}
    MONTH_MAP = {
        "January": 1, "February": 2, "March": 3, "April": 4, "May": 5,
        "June": 6, "July": 7, "August": 8, "September": 9, "October": 10,
        "November": 11, "December": 12,
    }

    CAP = ['Amount_invested_monthly', 'Monthly_Balance', 'Total_EMI_per_month', 'Outstanding_Debt', 'Delay_from_due_date', 'Annual_Income', 'Monthly_Inhand_Salary', 'Num_Credit_Inquiries', 'Changed_Credit_Limit', 'Num_of_Delayed_Payment', 'Num_of_Loan', 'Num_Credit_Card', 'Age']

    def __init__(self) -> None:
        self.fitted_ = False

        self.num_medians_: Dict[str, float] = {}
        self.cat_modes_: Dict[str, Any] = {}
        self.quantile_caps_: Dict[str, Tuple[float, float]] = {}
        self.loan_classes_: List[str] = []
        self.occupation_categories_: List[str] = []
        self.feature_columns_: List[str] = []

    @staticmethod
    def _strip_numeric(series: pd.Series) -> pd.Series:
        s = (series.astype(str).str.replace("_", "", regex=False).replace({"": np.nan, "nan": np.nan}))
        return pd.to_numeric(s, errors="coerce")

    @staticmethod
    def _hist_to_months(value: Any) -> float:
        if pd.isna(value):
            return np.nan
        y = re.search(r"(\d+)\s*Year", str(value))
        m = re.search(r"(\d+)\s*Month", str(value))
        return (int(y.group(1)) if y else 0) * 12 + (int(m.group(1)) if m else 0)

    @staticmethod
    def _parse_loan_types(value: Any) -> List[str]:
        if pd.isna(value):
            return []
        cleaned = re.sub(r",\s*and\s+", ", ", str(value))
        return list({t.strip() for t in cleaned.split(",") if t.strip()})

    def _basic_clean(self, df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()

        # strip junk characters out of numeric columns
        for c in self.NUMERIC_PLACEHOLDER_COLS:
            if c in df.columns:
                df[c] = self._strip_numeric(df[c])

        # replace sentinel placeholders with NaN
        replacements = {
            "Occupation": "_______", "Credit_Mix": "_",
            "Payment_of_Min_Amount": "NM", "Payment_Behaviour": "!@9#%8",
        }
        for col, bad in replacements.items():
            if col in df.columns:
                df[col] = df[col].replace(bad, np.nan)

        # impossible values to NaN
        if "Age" in df.columns:
            df.loc[(df["Age"] < 14) | (df["Age"] > 100), "Age"] = np.nan
        if "Num_of_Loan" in df.columns:
            df.loc[df["Num_of_Loan"] < 0, "Num_of_Loan"] = np.nan
        if "Num_Bank_Accounts" in df.columns:
            df.loc[df["Num_Bank_Accounts"] > 20, "Num_Bank_Accounts"] = np.nan
        if "Num_Credit_Card" in df.columns:
            df.loc[df["Num_Credit_Card"] > 20, "Num_Credit_Card"] = np.nan
        if "Interest_Rate" in df.columns:
            df.loc[df["Interest_Rate"] > 50, "Interest_Rate"] = np.nan
        if "Num_of_Delayed_Payment" in df.columns:
            df.loc[df["Num_of_Delayed_Payment"] < 0, "Num_of_Delayed_Payment"] = np.nan

        if "Credit_History_Age" in df.columns:
            df["Credit_History_Age"] = df["Credit_History_Age"].apply(self._hist_to_months)

        # within-customer forward/backward fill
        if ID_COL in df.columns:
            grp = df.groupby(ID_COL)
            for c in self.GROUP_IMPUTE_COLS:
                if c not in df.columns:
                    continue
                if df[c].dtype == object:
                    df[c] = grp[c].transform(lambda x: x.ffill().bfill())
                else:
                    df[c] = grp[c].transform(lambda x: x.fillna(x.median()))

            if "Type_of_Loan" in df.columns:
                df["Type_of_Loan"] = (grp["Type_of_Loan"]
                                      .transform(lambda x: x.ffill().bfill()))
        if "Type_of_Loan" in df.columns:
            df["Type_of_Loan"] = df["Type_of_Loan"].fillna("Not Specified")

        return df

    def _encode_target(self, df: pd.DataFrame) -> pd.DataFrame:
        if TARGET_COL in df.columns and df[TARGET_COL].dtype == object:
            df[TARGET_COL] = df[TARGET_COL].map(CREDIT_SCORE_MAP)
        return df

    def _encode_categoricals(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Credit_Mix" in df.columns and df["Credit_Mix"].dtype == object:
            df["Credit_Mix"] = df["Credit_Mix"].map(self.CREDIT_MIX_MAP)
        if "Payment_of_Min_Amount" in df.columns and df["Payment_of_Min_Amount"].dtype == object:
            df["Payment_of_Min_Amount"] = df["Payment_of_Min_Amount"].map({"No": 0, "Yes": 1})

        if "Payment_Behaviour" in df.columns:
            parts = df["Payment_Behaviour"].astype(str).str.split("_", n=2, expand=True)
            df["Spend_Level"] = parts[0].map(self.SPEND_MAP)
            df["Payment_Value"] = parts[2].str.split("_").str[0].map(self.VALUE_MAP)
            df = df.drop(columns=["Payment_Behaviour"])

        if "Month" in df.columns and df["Month"].dtype == object:
            df["Month"] = df["Month"].map(self.MONTH_MAP)
        return df

    def _expand_loan_flags(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Type_of_Loan" not in df.columns:
            return df
        loan_lists = df["Type_of_Loan"].apply(self._parse_loan_types)
        for cls in self.loan_classes_:
            col = f"Has_{cls.strip().replace(' ', '_').replace('-', '_')}"
            df[col] = loan_lists.apply(lambda lst: int(cls in lst))
        return df.drop(columns=["Type_of_Loan"])

    def _expand_occupation(self, df: pd.DataFrame) -> pd.DataFrame:
        if "Occupation" not in df.columns:
            return df
        for cat in self.occupation_categories_:
            df[f"Occupation_{cat}"] = (df["Occupation"] == cat).astype(bool)
        return df.drop(columns=["Occupation"])

    def _cap_outliers(self, df: pd.DataFrame) -> pd.DataFrame:
        for c in ("Delay_from_due_date", "Changed_Credit_Limit", "Num_Bank_Accounts"):
            if c in df.columns:
                df[c] = df[c].abs()
        for c, (lo, hi) in self.quantile_caps_.items():
            if c in df.columns:
                df[c] = df[c].clip(lower=lo, upper=hi)
        return df

    def fit(self, df: pd.DataFrame) -> "CreditDataPreprocessor":
        self.fit_transform(df)
        return self

    def fit_transform(self, df: pd.DataFrame) -> pd.DataFrame:
        logger.info("Preprocessor: fitting on %d rows", len(df))
        df = self._basic_clean(df)
        df = self._encode_target(df)

        # learn loan classes + occupations then they are dropped
        if "Type_of_Loan" in df.columns:
            all_types = set()
            df["Type_of_Loan"].apply(lambda v: all_types.update(self._parse_loan_types(v)))
            self.loan_classes_ = sorted(all_types)
        if "Occupation" in df.columns:
            occ = df["Occupation"].dropna()
            occ = occ[occ != "_______"]
            self.occupation_categories_ = sorted(occ.astype(str).str.replace(" ", "_").unique())
            df["Occupation"] = df["Occupation"].astype(str).str.replace(" ", "_")

        # remaining-null imputation: learn medians (numeric) and modes (categorical)
        df = df.drop(columns=[c for c in self.DROP_COLS if c in df.columns])
        df = self._encode_categoricals(df)

        feature_frame = df.drop(columns=[c for c in (TARGET_COL, ID_COL) if c in df.columns])
        for c in feature_frame.columns:
            if pd.api.types.is_numeric_dtype(feature_frame[c]):
                self.num_medians_[c] = float(feature_frame[c].median())
            else:
                mode = feature_frame[c].mode()
                self.cat_modes_[c] = mode.iloc[0] if not mode.empty else 0

        # winsorisation caps
        for c in self.CAP:
            if c in df.columns:
                self.quantile_caps_[c] = (float(df[c].quantile(0.05)),
                                          float(df[c].quantile(0.95)))

        self.fitted_ = True
        out = self._finalize(df)
        self.feature_columns_ = [c for c in out.columns if c not in (TARGET_COL, ID_COL)]
        logger.info("Preprocessor: produced %d features", len(self.feature_columns_))
        return out

    def transform(self, df: pd.DataFrame) -> pd.DataFrame:
        if not self.fitted_:
            raise RuntimeError("Preprocessor must be fitted before transform().")
        df = self._basic_clean(df)
        df = self._encode_target(df)
        if "Occupation" in df.columns:
            df["Occupation"] = df["Occupation"].astype(str).str.replace(" ", "_")
        df = df.drop(columns=[c for c in self.DROP_COLS if c in df.columns])
        df = self._encode_categoricals(df)
        out = self._finalize(df)
        # make sure identical feature columns/order as training
        for col in self.feature_columns_:
            if col not in out.columns:
                out[col] = 0
        keep = [c for c in (ID_COL, TARGET_COL) if c in out.columns] + self.feature_columns_
        return out[keep]

    def _finalize(self, df: pd.DataFrame) -> pd.DataFrame:
        df = self._expand_loan_flags(df)
        df = self._expand_occupation(df)

        # fill any remaining nulls using learned medians/modes
        for c, med in self.num_medians_.items():
            if c in df.columns:
                df[c] = pd.to_numeric(df[c], errors="coerce").fillna(med)
        for c, mode in self.cat_modes_.items():
            if c in df.columns:
                df[c] = df[c].fillna(mode)

        df = self._cap_outliers(df)
        return df

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Preprocessor saved -> %s", path)

    @staticmethod
    def load(path: str) -> "CreditDataPreprocessor":
        with open(path, "rb") as f:
            return pickle.load(f)


# Bundles everything needed to score a *raw* credit record in one object
class CreditScoreBundle:
    def __init__(self, preprocessor: "CreditDataPreprocessor", model: Any, feature_columns: List[str], label_names: Optional[List[str]] = None, model_name: str = "") -> None:
        self.preprocessor = preprocessor
        self.model = model
        self.feature_columns = list(feature_columns)
        self.label_names = list(label_names) if label_names is not None else list(LABEL_NAMES)
        self.model_name = model_name

    def _features(self, raw_df: pd.DataFrame) -> pd.DataFrame:
        out = self.preprocessor.transform(raw_df)
        return out[self.feature_columns]

    def predict(self, raw_df: pd.DataFrame):
        return self.model.predict(self._features(raw_df))

    def predict_proba(self, raw_df: pd.DataFrame):
        return self.model.predict_proba(self._features(raw_df))

    def predict_labels(self, raw_df: pd.DataFrame) -> List[str]:
        return [self.label_names[int(i)] for i in self.predict(raw_df)]

    def save(self, path: str) -> None:
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Bundle saved -> %s", path)

    @staticmethod
    def load(path: str) -> "CreditScoreBundle":
        with open(path, "rb") as f:
            return pickle.load(f)


# SPLITTING
class DataSplitter:
    def __init__(self, n_splits: int = 5, random_state: int = 42) -> None:
        self.n_splits = n_splits
        self.random_state = random_state

    def split(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.Series, pd.Series]:
        from sklearn.model_selection import StratifiedGroupKFold

        id_like = [c for c in ["ID", "Name", "SSN"] if c in df.columns]
        X = df.drop(columns=[TARGET_COL, ID_COL] + id_like)
        y = df[TARGET_COL].astype(int)
        groups = df[ID_COL]

        sgkf = StratifiedGroupKFold(n_splits=self.n_splits, shuffle=True, random_state=self.random_state)
        train_idx, test_idx = next(sgkf.split(X, y, groups=groups))

        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        overlap = set(groups.iloc[train_idx]) & set(groups.iloc[test_idx])
        assert not overlap, f"Leakage {len(overlap)} customers in both sets"

        logger.info("Split -> train=%d rows / test=%d rows", len(X_train), len(X_test))
        return (X_train.reset_index(drop=True), X_test.reset_index(drop=True), y_train.reset_index(drop=True), y_test.reset_index(drop=True))


# 3. MODELS
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
        """bee"""

    @abstractmethod
    def search_space(self, trial) -> Dict[str, Any]:
        """the"""

    @abstractmethod
    def default_params(self) -> Dict[str, Any]:
        """original"""

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
            core_idx, val_idx = train_test_split( np.arange(len(X)), test_size=0.15, stratify=y, random_state=self.random_state)
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




# EVALUATION
@dataclass
class EvalResult:
    macro_f1: float
    accuracy: float
    report: Dict[str, Any]
    confusion_path: Optional[str] = None
    metrics: Dict[str, float] = field(default_factory=dict)

# Computes metrics and renders a confusion-matrix image artifact.
class ModelEvaluator:
    def __init__(self, label_names: List[str] = LABEL_NAMES) -> None:
        self.label_names = label_names

    def evaluate(self, model, X_test, y_test, name: str, artifact_dir: str) -> EvalResult:
        from sklearn.metrics import (accuracy_score, classification_report,
                                     f1_score)

        y_pred = model.predict(X_test)
        macro_f1 = float(f1_score(y_test, y_pred, average="macro"))
        weighted_f1 = float(f1_score(y_test, y_pred, average="weighted"))
        acc = float(accuracy_score(y_test, y_pred))
        report = classification_report(
            y_test, y_pred, target_names=self.label_names, output_dict=True)

        metrics = {"macro_f1": macro_f1, "weighted_f1": weighted_f1, "accuracy": acc}
        for lbl in self.label_names:
            metrics[f"f1_{lbl}"] = float(report[lbl]["f1-score"])
            metrics[f"recall_{lbl}"] = float(report[lbl]["recall"])

        cm_path = self._plot_confusion(y_test, y_pred, name, artifact_dir)
        logger.info("[%s] test macro-F1=%.4f | accuracy=%.4f", name, macro_f1, acc)
        return EvalResult(macro_f1, acc, report, cm_path, metrics)

    def _plot_confusion(self, y_true, y_pred, name: str, artifact_dir: str) -> str:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
        from sklearn.metrics import ConfusionMatrixDisplay

        os.makedirs(artifact_dir, exist_ok=True)
        fig, ax = plt.subplots(figsize=(6, 5))
        ConfusionMatrixDisplay.from_predictions(
            y_true, y_pred, display_labels=self.label_names, ax=ax, colorbar=False)
        ax.set_title(f"{name} — Confusion Matrix (Test)")
        fig.tight_layout()
        path = os.path.join(artifact_dir, f"confusion_{name}.png")
        fig.savefig(path, dpi=120)
        plt.close(fig)
        return path


# End-to-end local training pipeline with MLflow experiment tracking.
class TrainingPipeline:
    def __init__(self,
                 data_path: str,
                 experiment_name: str = "credit_score_classification",
                 n_trials: int = 30,
                 artifact_dir: str = "artifacts",
                 tracking_uri: Optional[str] = None,
                 random_state: int = 42,
                 models: Optional[List[str]] = None,
                 use_smote: bool = True,
                 optuna_storage: Optional[str] = "sqlite:///optuna.db",
                 tune_n_jobs: int = 1,
                 override_best: bool = False) -> None:
        self.data_path = data_path
        self.experiment_name = experiment_name
        self.n_trials = n_trials
        self.artifact_dir = artifact_dir
        self.tracking_uri = tracking_uri or f"sqlite:///{os.path.abspath('mlflow.db')}"
        self.random_state = random_state
        self.models = [m.lower() for m in models] if models else None
        self.use_smote = use_smote
        self.override_best = override_best

        self.optuna_storage = optuna_storage
        self.tune_n_jobs = tune_n_jobs

        self.preprocessor = CreditDataPreprocessor()
        self.evaluator = ModelEvaluator()
        os.makedirs(self.artifact_dir, exist_ok=True)

    def _model_zoo(self) -> List[BaseModel]:
        all_models = [
            XGBModel(self.random_state),
            RandomForestModel(self.random_state),
            LightGBMModel(self.random_state),
            KNNModel(self.random_state),
            ExtraTreesModel(self.random_state),
            DecisionTreeModel(self.random_state),
        ]
        if not self.models:
            return all_models
        zoo = [m for m in all_models if m.name in self.models]
        unknown = set(self.models) - {m.name for m in all_models}
        if unknown:
            valid = ", ".join(m.name for m in all_models)
            raise ValueError(f"Unknown model(s): {unknown}. Valid names: {valid}")
        return zoo

    def run(self) -> Dict[str, Any]:
        import mlflow

        logger.info("Loading raw data: %s", self.data_path)
        raw = pd.read_csv(self.data_path, index_col=0)

        clean = self.preprocessor.fit_transform(raw)
        self.preprocessor.save(os.path.join(self.artifact_dir, "preprocessor.pkl"))
        with open(os.path.join(self.artifact_dir, "feature_columns.json"), "w") as f:
            json.dump(self.preprocessor.feature_columns_, f, indent=2)

        X_train, X_test, y_train, y_test = DataSplitter(
            random_state=self.random_state).split(clean)

        groups_train = self._train_groups(clean, len(X_train))

        smote = None
        if self.use_smote:
            from imblearn.over_sampling import SMOTE
            smote = SMOTE(sampling_strategy="auto", k_neighbors=5,
                          random_state=self.random_state)
            logger.info("SMOTE enabled (imblearn, k_neighbors=5, strategy=auto)")

        mlflow.set_tracking_uri(self.tracking_uri)
        mlflow.set_experiment(self.experiment_name)

        results: List[Tuple[str, float, str]] = []
        best = {"name": None, "macro_f1": -1.0, "model": None, "run_id": None, "model_uri": None}

        for model_def in self._model_zoo():
            with mlflow.start_run(run_name=model_def.name) as run:
                mlflow.set_tags({"model_type": model_def.name,
                                 "stage": "local_training"})
                mlflow.log_param("n_trials", self.n_trials)
                mlflow.log_param("requires_scaling", model_def.requires_scaling)
                mlflow.log_param("smote", self.use_smote)

                params = model_def.tune(X_train, y_train, groups_train,
                                        self.n_trials, smote=smote,
                                        n_jobs=self.tune_n_jobs,
                                        storage=self.optuna_storage)
                mlflow.log_params({f"hp_{k}": v for k, v in params.items()})
                self._log_optuna_plots(mlflow, model_def, self.artifact_dir)

                trained = model_def.fit_final(X_train, y_train, params,
                                              smote=smote, groups=groups_train)
                ev = self.evaluator.evaluate(
                    trained.model, X_test, y_test, model_def.name, self.artifact_dir)

                mlflow.log_metrics(ev.metrics)
                if ev.confusion_path:
                    mlflow.log_artifact(ev.confusion_path, artifact_path="plots")
                report_path = os.path.join(
                    self.artifact_dir, f"report_{model_def.name}.json")
                with open(report_path, "w") as f:
                    json.dump(ev.report, f, indent=2)
                mlflow.log_artifact(report_path, artifact_path="reports")

                model_uri = self._log_model(mlflow, model_def, trained.model)

                results.append((model_def.name, ev.macro_f1, run.info.run_id))
                if ev.macro_f1 > best["macro_f1"]:
                    best.update(name=model_def.name, macro_f1=ev.macro_f1,
                                model=trained.model, run_id=run.info.run_id,
                                model_uri=model_uri)

        self._register_best(mlflow, best)
        self._summary(results, best)
        return best

    @staticmethod
    def _train_groups(clean: pd.DataFrame, _n: int) -> pd.Series:
        from sklearn.model_selection import StratifiedGroupKFold
        id_like = [c for c in ["ID", "Name", "SSN"] if c in clean.columns]
        X = clean.drop(columns=[TARGET_COL, ID_COL] + id_like)
        y = clean[TARGET_COL].astype(int)
        groups = clean[ID_COL]
        sgkf = StratifiedGroupKFold(n_splits=5, shuffle=True, random_state=42)
        train_idx, _ = next(sgkf.split(X, y, groups=groups))
        return groups.iloc[train_idx].reset_index(drop=True)

    @staticmethod
    def _log_model(mlflow, model_def: BaseModel, model) -> Optional[str]:
        try:
            if model_def.name == "xgboost":
                import mlflow.xgboost
                info = mlflow.xgboost.log_model(model, name="model")
            elif model_def.name == "lightgbm":
                import mlflow.lightgbm
                info = mlflow.lightgbm.log_model(model, name="model")
            else:
                import mlflow.sklearn
                info = mlflow.sklearn.log_model(model, name="model")
            return info.model_uri
        except Exception as exc:  # logging a model must never crash training
            logger.warning("[%s] mlflow model logging failed: %s", model_def.name, exc)
            return None

    @staticmethod
    def _log_optuna_plots(mlflow, model_def: BaseModel, artifact_dir: str) -> None:
        study = getattr(model_def, "study_", None)
        if study is None:
            return
        try:
            import matplotlib
            matplotlib.use("Agg")
            import matplotlib.pyplot as plt
            from optuna.visualization.matplotlib import (
                plot_optimization_history, plot_param_importances)

            completed = [t for t in study.trials if t.state.name == "COMPLETE"]
            if len(completed) < 2:  # importance/history need a few finished trials
                return
            os.makedirs(artifact_dir, exist_ok=True)
            for plot_fn, label in ((plot_optimization_history, "history"),
                                   (plot_param_importances, "param_importance")):
                try:
                    ax = plot_fn(study)
                    fig = (ax.get_figure() if hasattr(ax, "get_figure")
                           else ax.flatten()[0].get_figure())
                    fig.tight_layout()
                    path = os.path.join(
                        artifact_dir, f"optuna_{label}_{model_def.name}.png")
                    fig.savefig(path, dpi=120)
                    plt.close(fig)
                    mlflow.log_artifact(path, artifact_path="optuna")
                except Exception as exc:
                    logger.warning("[%s] optuna %s plot skipped: %s",
                                   model_def.name, label, exc)
        except Exception as exc:
            logger.warning("[%s] optuna plotting unavailable: %s",
                           model_def.name, exc)

    def _register_best(self, mlflow, best: Dict[str, Any]) -> None:
        if best["model"] is None:
            return

        json_path = os.path.join(self.artifact_dir, "best_model.json")
        pkl_path = os.path.join(self.artifact_dir, "best_model.pkl")

        # Guard against overwriting a better champion (bypass with --override-best).
        if not self.override_best and os.path.exists(json_path):
            with open(json_path) as f:
                current_best = json.load(f)
            current_f1 = float(current_best.get("macro_f1", -1.0))
            if best["macro_f1"] <= current_f1:
                logger.info(
                    "Selective run best '%s' (macro-F1=%.4f) does NOT beat "
                    "persisted champion '%s' (macro-F1=%.4f) keeping existing artifacts.",
                    best["name"], best["macro_f1"],
                    current_best.get("name"), current_f1,
                )
                return
            logger.info(
                "Selective run best '%s' (macro-F1=%.4f) beats persisted champion "
                "'%s' (macro-F1=%.4f) — updating artifacts.",
                best["name"], best["macro_f1"],
                current_best.get("name"), current_f1,
            )

        bundle = CreditScoreBundle(
            preprocessor=self.preprocessor,
            model=best["model"],
            feature_columns=list(self.preprocessor.feature_columns_),
            label_names=LABEL_NAMES,
            model_name=best["name"],
        )
        bundle.save(pkl_path)
        with open(json_path, "w") as f:
            json.dump({"name": best["name"], "macro_f1": best["macro_f1"],
                       "run_id": best["run_id"]}, f, indent=2)
        try:
            model_uri = best.get("model_uri") or f"runs:/{best['run_id']}/model"
            mlflow.register_model(model_uri, "credit_score_best_model")
            logger.info("Registered best model '%s' in MLflow registry", best["name"])
        except Exception as exc:
            logger.warning("Model registry step skipped: %s", exc)

    @staticmethod
    def _summary(results: List[Tuple[str, float, str]], best: Dict[str, Any]) -> None:
        logger.info("=" * 56)
        logger.info("%-16s %-10s", "MODEL", "MACRO-F1")
        for name, f1, _ in sorted(results, key=lambda r: r[1], reverse=True):
            marker = "  <-- BEST" if name == best["name"] else ""
            logger.info("%-16s %.4f%s", name, f1, marker)
        logger.info("=" * 56)
        logger.info("Best model: %s (macro-F1=%.4f)", best["name"], best["macro_f1"])


# CLI
def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description="Local credit-score training pipeline (MLflow).")
    p.add_argument("--data", default="data_A.csv", help="Path to raw CSV.")
    p.add_argument("--experiment", default="credit_score_classification",
                   help="MLflow experiment name.")
    p.add_argument("--n-trials", type=int, default=30,
                   help="Optuna trials per model (0 = use defaults, fast).")
    p.add_argument("--artifact-dir", default="artifacts", help="Where to save outputs.")
    p.add_argument("--tracking-uri", default=None,
                   help="MLflow tracking URI (default: ./mlruns).")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument(
        "--models", nargs="+", default=None,
        metavar="NAME",
        help=(
            "One or more model names to train (default: all). "
            "Choices: xgboost random_forest lightgbm knn extra_trees decision_tree"
        ),
    )
    p.add_argument("--no-smote", action="store_true",
                   help="Disable SMOTE oversampling (enabled by default).")
    p.add_argument("--optuna-storage", default="sqlite:///optuna.db",
                   help=("Persistent Optuna study store so tuning resumes across "
                         "runs. Use 'none' to disable persistence."))
    p.add_argument("--tune-jobs", type=int, default=1,
                   help="Number of Optuna trials to run in parallel per model.")
    p.add_argument("--override-best", action="store_true",
                   help="Ignore the persisted champion and always overwrite artifacts.")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    optuna_storage = None if str(args.optuna_storage).lower() in ("none", "") \
        else args.optuna_storage
    pipeline = TrainingPipeline(
        data_path=args.data,
        experiment_name=args.experiment,
        n_trials=args.n_trials,
        artifact_dir=args.artifact_dir,
        tracking_uri=args.tracking_uri,
        random_state=args.seed,
        models=args.models,
        use_smote=not args.no_smote,
        optuna_storage=optuna_storage,
        tune_n_jobs=args.tune_jobs,
        override_best=args.override_best,
    )
    best = pipeline.run()
    logger.info("Done. Best=%s | artifacts in '%s' | MLflow UI: `mlflow ui --backend-store-uri sqlite:///mlflow.db`",
                best["name"], args.artifact_dir)


if __name__ == "__main__":
    main()
