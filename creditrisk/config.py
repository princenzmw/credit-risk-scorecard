"""Central configuration. Every assumption that affects results lives here."""
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DATA_PATH = ROOT / "data" / "raw" / "cs-training.csv"
MODELS_DIR = ROOT / "models"
REPORTS_DIR = ROOT / "reports"
FIGURES_DIR = REPORTS_DIR / "figures"

RANDOM_STATE = 42
TARGET = "SeriousDlqin2yrs"
INDEX_COL = "Unnamed: 0"

RAW_FEATURES = [
    "RevolvingUtilizationOfUnsecuredLines",
    "age",
    "NumberOfTime30-59DaysPastDueNotWorse",
    "DebtRatio",
    "MonthlyIncome",
    "NumberOfOpenCreditLinesAndLoans",
    "NumberOfTimes90DaysLate",
    "NumberRealEstateLoansOrLines",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfDependents",
]
PAST_DUE_COLS = [
    "NumberOfTime30-59DaysPastDueNotWorse",
    "NumberOfTime60-89DaysPastDueNotWorse",
    "NumberOfTimes90DaysLate",
]

# Split: 60 % train / 20 % validation / 20 % test (stratified on the target).
VAL_SIZE = 0.20
TEST_SIZE = 0.20

# Past-due columns use 96 and 98 as special codes (not real counts).
PAST_DUE_SENTINELS = (96, 98)
PAST_DUE_CAP = 10
CAP_QUANTILE = 0.999

# Business assumption (NOT a fact about the data): approving a loan that
# defaults costs COST_FN times as much as declining a loan that would repay.
COST_FN = 5.0
COST_FP = 1.0

# Champion rule: prefer the simpler model unless the challenger beats it on
# validation ROC-AUC by more than this margin.
CHALLENGER_MARGIN = 0.005

# Illustrative probability-of-default bands (cut-offs are a design choice).
PD_BAND_EDGES = [0.0, 0.03, 0.08, 0.20, 1.0000001]
PD_BAND_LABELS = ["Low", "Moderate", "High", "Very high"]
