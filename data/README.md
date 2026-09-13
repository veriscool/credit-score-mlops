# Dataset

This project uses a subset of a public **Kaggle "Credit Score Classification"** dataset
of synthetic customer financial records (25,000 rows). The data is not committed to
this repository because:

- the license/redistribution terms of the original Kaggle dataset aren't fully clear, and
- it contains columns (`Name`, `SSN`) that look like PII, even though the underlying
  records are synthetic.

## How to get the data

1. Search Kaggle for "Credit Score Classification" (multi-class, ~100k rows, columns
   including `Customer_ID`, `Month`, `Age`, `Occupation`, `Annual_Income`, `Credit_Mix`,
   `Payment_Behaviour`, `Credit_Score`, etc.).
2. Download the CSV and place it at `data/data_A.csv` (or pass `--data <path>` to
   `pipeline.py`).

## Expected columns

The preprocessing pipeline (`pipeline.py` / `CreditDataPreprocessor`) expects the raw
Kaggle schema — 23 input columns including `Customer_ID`, `Month`, `Age`, `Occupation`,
`Annual_Income`, `Monthly_Inhand_Salary`, `Num_Bank_Accounts`, `Num_Credit_Card`,
`Interest_Rate`, `Num_of_Loan`, `Type_of_Loan`, `Delay_from_due_date`,
`Num_of_Delayed_Payment`, `Changed_Credit_Limit`, `Num_Credit_Inquiries`, `Credit_Mix`,
`Outstanding_Debt`, `Credit_Utilization_Ratio`, `Credit_History_Age`,
`Payment_of_Min_Amount`, `Total_EMI_per_month`, `Amount_invested_monthly`,
`Payment_Behaviour`, `Monthly_Balance`, and the target `Credit_Score`
(`Poor` / `Standard` / `Good`). `ID`, `Name`, and `SSN` are dropped by the pipeline and
don't need to be present.
