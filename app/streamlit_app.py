"""Streamlit app: score one applicant with the saved model (no training on upload)."""
from __future__ import annotations

import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
import streamlit as st  # noqa: E402

from creditrisk import config as C  # noqa: E402
from creditrisk.explain import occlusion_contributions  # noqa: E402
from creditrisk.metrics import assign_band  # noqa: E402
from creditrisk.predict import load_artifacts  # noqa: E402

st.set_page_config(page_title="Credit-Risk Scorecard", page_icon="📊", layout="wide")

FRIENDLY = {
    "RevolvingUtilizationOfUnsecuredLines": "Credit-card utilisation",
    "age": "Age",
    "NumberOfTime30-59DaysPastDueNotWorse": "30-59 days late (count)",
    "DebtRatio": "Debt ratio",
    "MonthlyIncome": "Monthly income",
    "NumberOfOpenCreditLinesAndLoans": "Open credit lines",
    "NumberOfTimes90DaysLate": "90+ days late (count)",
    "NumberRealEstateLoansOrLines": "Real-estate loans",
    "NumberOfTime60-89DaysPastDueNotWorse": "60-89 days late (count)",
    "NumberOfDependents": "Dependents",
}


@st.cache_resource
def get_artifacts():
    return load_artifacts(C.MODELS_DIR)


def sidebar_inputs(ref: dict) -> pd.DataFrame:
    st.sidebar.header("Applicant")
    age = st.sidebar.number_input("Age", 18, 100, int(ref["age"]))
    util = st.sidebar.number_input(
        "Credit-card utilisation (1.0 = 100% of limit)", 0.0, 5.0, round(ref["RevolvingUtilizationOfUnsecuredLines"], 2), 0.01)
    debt = st.sidebar.number_input("Debt ratio (monthly debt / monthly income)", 0.0, 5.0, round(ref["DebtRatio"], 2), 0.01)
    income_unknown = st.sidebar.checkbox("Monthly income unknown")
    income = st.sidebar.number_input("Monthly income (dataset currency)", 0, 1_000_000, int(ref["MonthlyIncome"]), 100,
                                     disabled=income_unknown)
    dep_unknown = st.sidebar.checkbox("Number of dependents unknown")
    deps = st.sidebar.number_input("Dependents", 0, 15, int(ref["NumberOfDependents"]), disabled=dep_unknown)
    lines = st.sidebar.number_input("Open credit lines and loans", 0, 60, int(ref["NumberOfOpenCreditLinesAndLoans"]))
    re_loans = st.sidebar.number_input("Real-estate loans", 0, 30, int(ref["NumberRealEstateLoansOrLines"]))
    st.sidebar.subheader("Payment history (last 2 years)")
    l3059 = st.sidebar.number_input("Times 30-59 days late", 0, 20, 0)
    l6089 = st.sidebar.number_input("Times 60-89 days late", 0, 20, 0)
    l90 = st.sidebar.number_input("Times 90+ days late", 0, 20, 0)
    return pd.DataFrame([{
        "RevolvingUtilizationOfUnsecuredLines": util, "age": age,
        "NumberOfTime30-59DaysPastDueNotWorse": l3059, "DebtRatio": debt,
        "MonthlyIncome": np.nan if income_unknown else income,
        "NumberOfOpenCreditLinesAndLoans": lines, "NumberOfTimes90DaysLate": l90,
        "NumberRealEstateLoansOrLines": re_loans, "NumberOfTime60-89DaysPastDueNotWorse": l6089,
        "NumberOfDependents": np.nan if dep_unknown else deps,
    }])


def score_tab(art: dict) -> None:
    row = sidebar_inputs(art["reference"])
    model = art["model"]
    default_t = float(art["threshold"]["threshold"])
    threshold = st.slider("Decision threshold (decline / review at or above)", 0.01, 0.80, default_t, 0.01,
                          help=f"Default {default_t:.2f} minimises cost assuming a missed default costs "
                               f"{art['threshold']['cost_fn']:g}x a wrongly declined good borrower.")
    pd_hat = float(model.predict_proba(row[C.RAW_FEATURES])[0, 1])
    band = str(assign_band([pd_hat]).iloc[0])
    c1, c2, c3 = st.columns(3)
    c1.metric("Probability of default (2 yrs)", f"{pd_hat:.1%}")
    c2.metric("Risk band", band)
    c3.metric("Decision at threshold", "Decline / review" if pd_hat >= threshold else "Approve")

    st.subheader("What is driving this score?")
    contrib = occlusion_contributions(model, row, art["reference"]).head(6)
    labels = [FRIENDLY[i] for i in contrib.index][::-1]
    values = contrib.values[::-1] * 100
    fig, ax = plt.subplots(figsize=(7, 3.2))
    ax.barh(labels, values, color=["#D9534F" if v > 0 else "#2E8B7A" for v in values])
    ax.axvline(0, color="grey", lw=0.8)
    ax.set_xlabel("Change in default probability vs a typical applicant (percentage points)")
    fig.tight_layout()
    st.pyplot(fig)
    st.caption("Each bar shows how the predicted risk changes if that one input is replaced by the training "
               "median. Red raises risk, green lowers it. Interactions are ignored, so this is a sensitivity "
               "view, not an exact decomposition.")


def performance_tab(art: dict) -> None:
    m = art["metrics"]
    t = m["test"]
    st.markdown(f"**Champion model:** {m['champion']}  \n**Evaluated once on a held-out test set of {m['n_rows']['test']:,} borrowers.**")
    a, b, c, d = st.columns(4)
    a.metric("ROC-AUC", f"{t['roc_auc']:.3f}")
    b.metric("PR-AUC", f"{t['pr_auc']:.3f}", help=f"A no-skill model scores about {t['base_rate']:.3f}")
    c.metric("KS", f"{t['ks']:.3f}")
    d.metric("Brier score", f"{t['brier']:.4f}")
    st.subheader("Do the risk bands separate borrowers?")
    bands = pd.DataFrame(m["bands_test"]).rename(columns={"band": "Band", "loans": "Loans", "observed_default_rate": "Observed default rate"})
    bands["Observed default rate"] = bands["Observed default rate"].map("{:.1%}".format)
    st.dataframe(bands, hide_index=True)
    fig_dir = C.FIGURES_DIR
    cols = st.columns(2)
    for col, name in zip(cols, ["calibration_test.png", "permutation_importance.png"]):
        if (fig_dir / name).exists():
            col.image(str(fig_dir / name))


def about_tab() -> None:
    card = C.REPORTS_DIR / "model_card.md"
    st.markdown(card.read_text(encoding="utf-8") if card.exists() else "Model card not found.")


def main() -> None:
    st.title("Credit-Risk Scorecard")
    st.caption("Estimates the probability that a borrower becomes 90+ days past due within two years. "
               "Trained on the 2011 Give Me Some Credit competition data. A portfolio project, not lending advice.")
    try:
        art = get_artifacts()
    except FileNotFoundError:
        st.error("Model files not found. Run `python -m creditrisk.train` first (see README).")
        return
    t1, t2, t3 = st.tabs(["Score an applicant", "Model performance", "Model card"])
    with t1:
        score_tab(art)
    with t2:
        performance_tab(art)
    with t3:
        about_tab()


main()
