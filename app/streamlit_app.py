"""
streamlit_app.py - local Streamlit web app for the credit-score classifier.

Run from the repo root:
    streamlit run app/streamlit_app.py

23-field input form, three sidebar presets (Poor / Standard / Good), and a
prediction card with per-class probability bars. Loads artifacts/best_model.pkl
(a CreditScoreBundle); requires pipeline.py (at the repo root) to be importable.
"""
from __future__ import annotations

import pickle
import sys
from pathlib import Path

import pandas as pd
import streamlit as st

# Locate the project root (this file lives in app/) so pipeline.py is importable.
ROOT = Path(__file__).resolve().parents[1]
if str(ROOT) not in sys.path:
    sys.path.insert(0, str(ROOT))

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
MONTHS = [
    "January", "February", "March", "April", "May", "June",
    "July", "August", "September", "October", "November", "December",
]

OCCUPATIONS = [
    "Accountant", "Architect", "Developer", "Doctor", "Engineer",
    "Entrepreneur", "Journalist", "Lawyer", "Manager", "Mechanic",
    "Media Manager", "Musician", "Scientist", "Teacher", "Writer",
]

LOAN_TYPES = [
    "Auto Loan", "Credit-Builder Loan", "Debt Consolidation Loan",
    "Home Equity Loan", "Mortgage Loan", "Not Specified",
    "Payday Loan", "Personal Loan", "Student Loan",
]

PAYMENT_BEHAVIOUR_OPTIONS = {
    "Low Spend, Small Value":   "Low_spent_Small_value_payments",
    "Low Spend, Medium Value":  "Low_spent_Medium_value_payments",
    "Low Spend, Large Value":   "Low_spent_Large_value_payments",
    "High Spend, Small Value":  "High_spent_Small_value_payments",
    "High Spend, Medium Value": "High_spent_Medium_value_payments",
    "High Spend, Large Value":  "High_spent_Large_value_payments",
}

LABEL_COLORS = {"Poor": "#ef4444", "Standard": "#f97316", "Good": "#22c55e"}
LABEL_BG     = {"Poor": "#fef2f2", "Standard": "#fff7ed", "Good": "#f0fdf4"}
LABEL_DESCRIPTIONS = {
    "Poor":     "High credit risk. Significant issues with payment history, "
                "high outstanding debt, or limited credit history.",
    "Standard": "Moderate credit risk. Some room for improvement in payment "
                "consistency and debt management.",
    "Good":     "Low credit risk. Strong payment history, manageable debt, "
                "and healthy credit practices.",
}

# ---------------------------------------------------------------------------
# Presets — one representative profile per class
# ---------------------------------------------------------------------------
PRESETS: dict[str, dict] = {
    "Poor": {
        "month": "January",   "age": 28,  "occupation": "Developer",
        "annual_income": 15000.0,          "monthly_salary": 1050.0,
        "num_bank_accounts": 9,            "num_credit_card": 8,
        "interest_rate": 34,               "num_of_loan": 7,
        "credit_mix": "Bad",
        "outstanding_debt": 3500.0,        "credit_utilization": 42.0,
        "ch_years": 3,                     "ch_months_val": 2,
        "num_inquiries": 17.0,
        "delay_from_due": 62,              "num_delayed": 22,
        "payment_min": "Yes",              "changed_limit": 0.5,
        "pay_behaviour_disp": "High Spend, Small Value",
        "total_emi": 350.0,                "amount_invested": 10.0,
        "monthly_balance": 50.0,
        "loan_list": ["Payday Loan", "Personal Loan"],
    },
    "Standard": {
        "month": "June",      "age": 35,  "occupation": "Engineer",
        "annual_income": 48000.0,          "monthly_salary": 3600.0,
        "num_bank_accounts": 4,            "num_credit_card": 4,
        "interest_rate": 14,               "num_of_loan": 3,
        "credit_mix": "Standard",
        "outstanding_debt": 1200.0,        "credit_utilization": 30.0,
        "ch_years": 10,                    "ch_months_val": 4,
        "num_inquiries": 5.0,
        "delay_from_due": 18,              "num_delayed": 6,
        "payment_min": "No",               "changed_limit": 8.0,
        "pay_behaviour_disp": "Low Spend, Medium Value",
        "total_emi": 120.0,                "amount_invested": 200.0,
        "monthly_balance": 350.0,
        "loan_list": ["Personal Loan", "Auto Loan"],
    },
    "Good": {
        "month": "September", "age": 45,  "occupation": "Engineer",
        "annual_income": 120000.0,         "monthly_salary": 9500.0,
        "num_bank_accounts": 2,            "num_credit_card": 3,
        "interest_rate": 7,                "num_of_loan": 2,
        "credit_mix": "Good",
        "outstanding_debt": 200.0,         "credit_utilization": 18.0,
        "ch_years": 22,                    "ch_months_val": 8,
        "num_inquiries": 1.0,
        "delay_from_due": 2,               "num_delayed": 0,
        "payment_min": "No",               "changed_limit": 15.0,
        "pay_behaviour_disp": "Low Spend, Large Value",
        "total_emi": 55.0,                 "amount_invested": 900.0,
        "monthly_balance": 2500.0,
        "loan_list": ["Mortgage Loan", "Auto Loan"],
    },
}


def _init_session_state() -> None:
    """Seed session state with Standard preset on first load."""
    if "_initialized" not in st.session_state:
        for k, v in PRESETS["Standard"].items():
            st.session_state[k] = v
        st.session_state["_initialized"] = True


def _load_preset(label: str) -> None:
    for k, v in PRESETS[label].items():
        st.session_state[k] = v
    st.session_state["_active_preset"] = label


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------
class _PipelineUnpickler(pickle.Unpickler):
    _REMAP = {"CreditScoreBundle", "CreditDataPreprocessor"}

    def find_class(self, module, name):
        if module == "__main__" and name in self._REMAP:
            import pipeline
            return getattr(pipeline, name)
        return super().find_class(module, name)


@st.cache_resource(show_spinner="Loading model...")
def load_bundle():
    bundle_path = ROOT / "artifacts" / "best_model.pkl"
    if not bundle_path.exists():
        st.error(f"Model file not found: {bundle_path}")
        st.stop()
    with open(bundle_path, "rb") as f:
        bundle = _PipelineUnpickler(f).load()
    if hasattr(bundle.model, "set_params"):
        try:
            bundle.model.set_params(device="cpu")
        except Exception:
            pass
    return bundle


def format_loan_types(selected: list[str]) -> str:
    if not selected:
        return "Not Specified"
    if len(selected) == 1:
        return selected[0]
    return ", ".join(selected[:-1]) + ", and " + selected[-1]


def run_prediction(bundle, record: dict):
    df = pd.DataFrame([record])
    labels = bundle.predict_labels(df)
    probas = bundle.predict_proba(df)
    prob_dict = {bundle.label_names[j]: float(p) for j, p in enumerate(probas[0])}
    return labels[0], prob_dict


# ---------------------------------------------------------------------------
# App bootstrap
# ---------------------------------------------------------------------------
st.set_page_config(
    page_title="Credit Score Predictor",
    page_icon="📊",
    layout="wide",
    initial_sidebar_state="expanded",
)

_init_session_state()

# Injecting subtle global styling updates
st.markdown(
    """
    <style>
    .stForm {
        border: 1px solid #e2e8f0 !important;
        border-radius: 12px !important;
        padding: 2rem !important;
        box-shadow: 0 4px 6px -1px rgba(0, 0, 0, 0.05), 0 2px 4px -1px rgba(0, 0, 0, 0.03);
    }
    div[data-testid="stExpander"] {
        border: 1px solid #e2e8f0 !important;
        border-radius: 8px !important;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# ---------------------------------------------------------------------------
# Sidebar
# ---------------------------------------------------------------------------
with st.sidebar:
    st.markdown(
        """
        <div style="margin-bottom: 15px;">
            <h2 style="margin: 0; font-size: 24px; font-weight: 700; color: #0cf2df;">Credit Score AI</h2>
            <p style="margin: 0; font-size: 14px; color: #64748b; font-weight: 500;">Model Deployment Suite</p>
        </div>
        """,
        unsafe_allow_html=True
    )
    st.caption("BINUS University · 2025/2026")
    st.divider()

    try:
        bundle = load_bundle()
        
        # Styled model status block
        st.markdown(
            f"""
            <div style="background-color: #f8fafc; border: 1px solid #e2e8f0; border-radius: 8px; padding: 12px; margin-bottom: 15px;">
                <span style="color: #22c55e; font-weight: bold; font-size: 14px;">● Model Operational</span><br>
                <small style="color: #64748b;">Algorithm:</small> <strong style="color: #0f172a;">{bundle.model_name.upper()}</strong><br>
                <small style="color: #64748b;">Features mapped:</small> <strong style="color: #0f172a;">{len(bundle.feature_columns)}</strong>
            </div>
            """, 
            unsafe_allow_html=True
        )
    except Exception as exc:
        st.error(f"Model load error: {exc}")
        st.stop()

    st.divider()

    # ── Preset loader ──────────────────────────────────────────────────────
    st.markdown("💡 **Quick Profiles**")
    st.caption("Populate form mockups with baseline classification archetypes:")

    col_p, col_s, col_g = st.columns(3)

    if col_p.button("Poor", use_container_width=True, help="High-risk customer profile"):
        _load_preset("Poor")
        st.rerun()

    if col_s.button("Standard", use_container_width=True, help="Mid-risk customer profile"):
        _load_preset("Standard")
        st.rerun()

    if col_g.button("Good", use_container_width=True, help="Low-risk customer profile"):
        _load_preset("Good")
        st.rerun()

    active = st.session_state.get("_active_preset", "")
    if active:
        color = LABEL_COLORS[active]
        st.markdown(
            f"""
            <div style='text-align: center; margin-top: 10px; background: {LABEL_BG[active]}; border: 1px solid {color}; border-radius: 6px; padding: 4px;'>
                <span style='font-size: 12px; color: {color}; font-weight: 600;'>Active Preset: {active}</span>
            </div>
            """,
            unsafe_allow_html=True,
        )

# ---------------------------------------------------------------------------
# Main content
# ---------------------------------------------------------------------------
st.title("📊 Customer Credit Score Predictor")
st.markdown(
    "Evaluate consumer risk profiling using predictive intelligence. "
    "Fill out the financial health parameters below to gauge probability boundaries."
)

# ---------------------------------------------------------------------------
# Input form
# ---------------------------------------------------------------------------
with st.form("credit_form", clear_on_submit=False):

    # ── Personal & Employment ──────────────────────────────────────────────
    st.markdown("### 👤 Personal & Employment")
    c1, c2, c3 = st.columns(3)
    month      = c1.selectbox("Assessment Month", MONTHS, key="month")
    age        = c2.number_input("Age", min_value=14, max_value=100, step=1, key="age")
    occupation = c3.selectbox("Occupation Category", OCCUPATIONS, key="occupation")

    st.divider()

    # ── Income & Financial Snapshot ────────────────────────────────────────
    st.markdown("### 💰 Financial Liquidity & Assets")
    c1, c2, c3 = st.columns(3)
    annual_income  = c1.number_input("Gross Annual Income ($)", min_value=0.0, step=500.0, format="%.2f", key="annual_income")
    monthly_salary = c2.number_input("Net Monthly Inhand Salary ($)", min_value=0.0, step=100.0, format="%.2f", key="monthly_salary")
    monthly_balance = c3.number_input("Residual Monthly Balance ($)", step=10.0, format="%.2f", key="monthly_balance")

    st.divider()

    # ── Accounts & Credit ─────────────────────────────────────────────────
    st.markdown("### 💳 Credit Architecture")
    c1, c2, c3, c4 = st.columns(4)
    num_bank_accounts = c1.number_input("Open Bank Accounts", min_value=0, max_value=20, step=1, key="num_bank_accounts")
    num_credit_card   = c2.number_input("Active Credit Cards", min_value=0, max_value=20, step=1, key="num_credit_card")
    interest_rate     = c3.number_input("Average Interest Rate (%)", min_value=0, max_value=50, step=1, key="interest_rate")
    num_of_loan       = c4.number_input("Total Active Loans", min_value=0, step=1, key="num_of_loan")

    c1, c2, c3 = st.columns(3)
    credit_mix         = c1.selectbox("Credit Mix Quality", ["Bad", "Standard", "Good"], key="credit_mix")
    outstanding_debt   = c2.number_input("Total Outstanding Debt ($)", min_value=0.0, step=50.0, format="%.2f", key="outstanding_debt")
    credit_utilization = c3.number_input("Credit Utilization Ratio (%)", min_value=0.0, max_value=100.0, step=0.5, format="%.2f", key="credit_utilization")

    st.divider()

    # ── Credit History & Payment Behaviour ─────────────────────────────────
    st.markdown("### ⏳ Credit History & Payment Behavior")
    
    with st.container(border=True):
        st.caption("Credit Bureau History Age")
        ch_col1, ch_col2 = st.columns(2)
        ch_years      = ch_col1.number_input("Years", min_value=0, max_value=50, step=1, key="ch_years")
        ch_months_val = ch_col2.number_input("Months", min_value=0, max_value=11, step=1, key="ch_months_val")

    c1, c2, c3 = st.columns(3)
    num_inquiries = c1.number_input("Hard Credit Inquiries (Last 12M)", min_value=0.0, step=1.0, key="num_inquiries")
    delay_from_due = c2.number_input("Avg Delay from Due Date (Days)", min_value=0, step=1, key="delay_from_due")
    num_delayed    = c3.number_input("Total Number of Delayed Payments", min_value=0, step=1, key="num_delayed")

    c1, c2 = st.columns(2)
    payment_min        = c1.selectbox("Frequently Pays Minimum Balance Only?", ["No", "Yes"], key="payment_min")
    pay_behaviour_disp = c2.selectbox("Discretionary Spending Pattern", list(PAYMENT_BEHAVIOUR_OPTIONS.keys()), key="pay_behaviour_disp")

    st.divider()

    # ── Obligations & Commitments ─────────────────────────────────────────
    st.markdown("### 🎯 Monthly Commitments & Portfolios")
    c1, c2 = st.columns(2)
    total_emi       = c1.number_input("Total Monthly Debt Installments / EMI ($)", min_value=0.0, step=10.0, format="%.2f", key="total_emi")
    amount_invested = c2.number_input("Monthly Outgoing Investments ($)", min_value=0.0, step=10.0, format="%.2f", key="amount_invested")

    loan_list = st.multiselect(
        "Current Open Loan Portfolios",
        LOAN_TYPES,
        key="loan_list",
    )

    st.markdown("<br>", unsafe_allow_html=True)
    submitted = st.form_submit_button(
        "⚡ Predict Credit Score",
        type="primary",
        use_container_width=True,
    )

# ---------------------------------------------------------------------------
# Prediction results
# ---------------------------------------------------------------------------
if submitted:
    record = {
        "Month":                    month,
        "Age":                      age,
        "Occupation":               occupation,
        "Annual_Income":            annual_income,
        "Monthly_Inhand_Salary":    monthly_salary,
        "Num_Bank_Accounts":        num_bank_accounts,
        "Num_Credit_Card":          num_credit_card,
        "Interest_Rate":            interest_rate,
        "Num_of_Loan":              num_of_loan,
        "Delay_from_due_date":      delay_from_due,
        "Num_of_Delayed_Payment":   num_delayed,
        "Changed_Credit_Limit":     changed_limit if 'changed_limit' in locals() else st.session_state.get('changed_limit', 0.0),
        "Num_Credit_Inquiries":     num_inquiries,
        "Credit_Mix":               credit_mix,
        "Outstanding_Debt":         outstanding_debt,
        "Credit_Utilization_Ratio": credit_utilization,
        "Credit_History_Age":       f"{ch_years} Years and {ch_months_val} Months",
        "Payment_of_Min_Amount":    payment_min,
        "Total_EMI_per_month":      total_emi,
        "Amount_invested_monthly":  amount_invested,
        "Payment_Behaviour":        PAYMENT_BEHAVIOUR_OPTIONS[pay_behaviour_disp],
        "Monthly_Balance":          monthly_balance,
        "Type_of_Loan":             format_loan_types(loan_list),
    }

    with st.spinner("Running inference engine pipeline..."):
        try:
            prediction, prob_dict = run_prediction(bundle, record)
        except Exception as exc:
            st.error(f"Inference error: {exc}")
            st.stop()

    st.markdown("<br><br>", unsafe_allow_html=True)
    st.subheader("📋 Evaluation Diagnostic Result")

    col_result, col_probs = st.columns([1, 1], gap="large")

    with col_result:
        color = LABEL_COLORS[prediction]
        bg    = LABEL_BG[prediction]
        desc  = LABEL_DESCRIPTIONS[prediction]
        st.markdown(
            f"""
            <div style="
                background:{bg};
                border: 1px solid {color}40;
                border-left:8px solid {color};
                border-radius:12px;
                padding:32px 24px;
                text-align:center;
                box-shadow: 0 10px 15px -3px rgba(0, 0, 0, 0.02);
            ">
                <p style="margin:0;font-size:14px;color:#64748b;text-transform:uppercase;
                          letter-spacing:1.5px;font-weight:600;">Predicted Tier</p>
                <p style="margin:12px 0 8px;font-size:56px;font-weight:800;
                          color:{color};line-height:1;letter-spacing:-1px;">{prediction}</p>
                <p style="margin:16px 0 0;font-size:14px;color:#334155;
                          line-height:1.6;font-weight: 400;">{desc}</p>
            </div>
            """,
            unsafe_allow_html=True,
        )

    with col_probs:
        st.markdown("<p style='font-weight:600; color:#475569; margin-bottom:16px;'>Confidence Distribution Bound</p>", unsafe_allow_html=True)
        for label, p in prob_dict.items():
            c   = LABEL_COLORS[label]
            pct = p * 100
            st.markdown(
                f"""
                <div style="margin-bottom:18px;">
                    <div style="display:flex;justify-content:between;margin-bottom:6px;">
                        <span style="font-weight:600;color:{c};flex:1;">{label} Status</span>
                        <span style="font-weight:700;color:#1e293b;">{pct:.1f}%</span>
                    </div>
                    <div style="background:#f1f5f9;border-radius:100px;
                                height:12px;overflow:hidden;width:100%;">
                        <div style="width:{pct:.1f}%;background:{c};
                                    height:100%;border-radius:100px;
                                    transition: width 0.6s ease;"></div>
                    </div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)
    with st.expander("🔍 Audit Trail Summary (Raw Feature Array JSON)"):
        st.json(record)