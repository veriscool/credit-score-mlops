"""Load, clean, and split the credit-score dataset.

Extracted from the project's local ../../pipeline.py so the cloud build uses
the exact same preprocessing as the local champion in
../../artifacts/best_model.pkl.
"""

from __future__ import annotations

import logging
import os
import re
from typing import Any, Dict, List, Tuple

import numpy as np
import pandas as pd

logger = logging.getLogger("data")

# Label encoding
TARGET_COL = "Credit_Score"
ID_COL = "Customer_ID"
LABEL_NAMES = ["Poor", "Standard", "Good"]
CREDIT_SCORE_MAP = {"Poor": 0, "Standard": 1, "Good": 2}


def load_dataset(path: str) -> pd.DataFrame:
    """Load the raw credit-score CSV (first column is a row index)."""
    return pd.read_csv(path, index_col=0)


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
        import pickle
        os.makedirs(os.path.dirname(path) or ".", exist_ok=True)
        with open(path, "wb") as f:
            pickle.dump(self, f)
        logger.info("Preprocessor saved -> %s", path)

    @staticmethod
    def load(path: str) -> "CreditDataPreprocessor":
        import pickle
        with open(path, "rb") as f:
            return pickle.load(f)


class DataSplitter:
    """Leakage-safe split: no Customer_ID appears in both train and test."""

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
        return (X_train.reset_index(drop=True), X_test.reset_index(drop=True),
                y_train.reset_index(drop=True), y_test.reset_index(drop=True))

    @staticmethod
    def train_groups(clean: pd.DataFrame, splitter: "DataSplitter") -> pd.Series:
        """Recover the Customer_ID series aligned to the train split above,
        needed as the `groups` argument for tuning/early-stopping."""
        from sklearn.model_selection import StratifiedGroupKFold
        id_like = [c for c in ["ID", "Name", "SSN"] if c in clean.columns]
        X = clean.drop(columns=[TARGET_COL, ID_COL] + id_like)
        y = clean[TARGET_COL].astype(int)
        groups = clean[ID_COL]
        sgkf = StratifiedGroupKFold(n_splits=splitter.n_splits, shuffle=True,
                                    random_state=splitter.random_state)
        train_idx, _ = next(sgkf.split(X, y, groups=groups))
        return groups.iloc[train_idx].reset_index(drop=True)
