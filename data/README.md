# Data

The raw data is **not** committed to this repository.

1. Open the Kaggle competition page: https://www.kaggle.com/competitions/GiveMeSomeCredit
2. Accept the competition rules (Kaggle requires a free account), then download `cs-training.csv` from the Data tab.
3. Place it at `data/raw/cs-training.csv`.

Columns: `SeriousDlqin2yrs` (target: 90+ days past due within two years), plus ten borrower features
(utilisation of credit lines, age, three past-due counts, debt ratio, monthly income, open credit lines,
real-estate loans, dependents). An unnamed first column is an index and is dropped by the loader.

Please check the competition's rules page before sharing the file anywhere.
