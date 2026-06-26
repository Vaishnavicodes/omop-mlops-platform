# omop-mlops-platform

Shared OMOP-CDM feature engineering and MLOps utilities for a healthcare ML platform. Provides
cohort building, feature extraction from standardized OMOP tables, and MLflow experiment
configuration used by two downstream apps: **ade-detection** and **trial-cohort**.

## Install

Pin to a release tag for reproducibility:

```bash
pip install git+https://github.com/Vaishnavicodes/omop-mlops-platform.git@v0.1.0
```

## Usage

```python
from omop_mlops.features import make_feature_matrix

feature_df = make_feature_matrix(
    person_df=person_df,
    condition_df=condition_df,
    drug_df=drug_df,
    measurement_df=measurement_df,
    index_dates=index_dates,   # pd.Series keyed by person_id
    lookback_days=365,
)
```

## Dependent apps

| App | Purpose |
|-----|---------|
| `ade-detection` | Adverse drug event detection model |
| `trial-cohort` | Clinical trial cohort selection pipeline |

Both apps install this library via the pinned git tag above.

## Development

```bash
git clone https://github.com/Vaishnavicodes/omop-mlops-platform.git
cd omop-mlops-platform
pip install -e .[dev]
pytest -q
ruff check .
black .
```
