"""Tests for temporal validity and ICD code normalization in omop_mlops.features."""

import pandas as pd
import pytest

from omop_mlops.features import (
    aggregate_measurements,
    comorbidity_flags,
    make_feature_matrix,
    normalize_icd,
)

# ---------------------------------------------------------------------------
# normalize_icd
# ---------------------------------------------------------------------------


@pytest.mark.parametrize(
    "raw, expected",
    [("250.0", "2500"), ("25000", "25000"), (" 410.9 ", "4109"), ("e930.1", "E9301")],
)
def test_normalize_icd_strips_dots_and_whitespace(raw, expected):
    assert normalize_icd(raw) == expected


# ---------------------------------------------------------------------------
# ICD format: dotted mapping must match MIMIC's undotted codes
# ---------------------------------------------------------------------------


def test_dotted_prefix_matches_mimic_undotted_code():
    person_ids = [1, 2]
    condition_df = pd.DataFrame(
        {
            "person_id": [1, 2],
            "condition_source_value": ["25000", "41401"],  # MIMIC-style, no dots
        }
    )
    mapping = {"diabetes": ["250.0"], "cad": ["414.0"]}
    result = comorbidity_flags(condition_df, person_ids, mapping=mapping).set_index("person_id")
    assert bool(result.loc[1, "diabetes"]) is True
    assert bool(result.loc[2, "diabetes"]) is False
    assert bool(result.loc[2, "cad"]) is True


# ---------------------------------------------------------------------------
# Temporal validity: comorbidities
# ---------------------------------------------------------------------------


def test_comorbidity_excludes_conditions_after_index_date():
    condition_df = pd.DataFrame(
        {
            "person_id": [1, 2],
            "condition_source_value": ["410", "410"],
            "condition_start_date": pd.to_datetime(["2020-06-01", "2019-06-01"]),
        }
    )
    index_dates = pd.Series({1: pd.Timestamp("2020-01-01"), 2: pd.Timestamp("2020-01-01")})
    result = comorbidity_flags(
        condition_df, [1, 2], mapping={"mi": ["410"]}, index_dates=index_dates
    ).set_index("person_id")
    assert bool(result.loc[1, "mi"]) is False  # diagnosed after the index date -> leakage
    assert bool(result.loc[2, "mi"]) is True


def test_comorbidity_lookback_window_excludes_old_conditions():
    condition_df = pd.DataFrame(
        {
            "person_id": [1],
            "condition_source_value": ["410"],
            "condition_start_date": pd.to_datetime(["2015-01-01"]),
        }
    )
    index_dates = pd.Series({1: pd.Timestamp("2020-01-01")})
    result = comorbidity_flags(
        condition_df,
        [1],
        mapping={"mi": ["410"]},
        index_dates=index_dates,
        lookback_days=365,
    ).set_index("person_id")
    assert bool(result.loc[1, "mi"]) is False


def test_comorbidity_with_index_dates_requires_start_date_column():
    condition_df = pd.DataFrame({"person_id": [1], "condition_source_value": ["410"]})
    with pytest.raises(KeyError):
        comorbidity_flags(
            condition_df,
            [1],
            mapping={"mi": ["410"]},
            index_dates=pd.Series({1: pd.Timestamp("2020-01-01")}),
        )


# ---------------------------------------------------------------------------
# Temporal validity: measurements
# ---------------------------------------------------------------------------


def test_measurements_after_index_date_are_excluded():
    measurement_df = pd.DataFrame(
        {
            "person_id": [1, 1],
            "measurement_concept_id": [3004249, 3004249],
            "value_as_number": [120.0, 999.0],
            "measurement_date": pd.to_datetime(["2020-01-05", "2020-03-01"]),
        }
    )
    index_dates = pd.Series({1: pd.Timestamp("2020-02-01")})
    result = aggregate_measurements(
        measurement_df, [1], agg=("max",), index_dates=index_dates
    ).set_index("person_id")
    assert result.loc[1, "3004249_max"] == 120.0  # the 999.0 reading is after the index date


def test_measurements_require_date_column_when_windowed():
    measurement_df = pd.DataFrame(
        {"person_id": [1], "measurement_concept_id": [1], "value_as_number": [1.0]}
    )
    with pytest.raises(KeyError):
        aggregate_measurements(
            measurement_df, [1], index_dates=pd.Series({1: pd.Timestamp("2020-01-01")})
        )


# ---------------------------------------------------------------------------
# End to end: feature matrix respects the index date for every feature family
# ---------------------------------------------------------------------------


def test_feature_matrix_has_no_post_index_information():
    person_df = pd.DataFrame({"person_id": [1]})
    condition_df = pd.DataFrame(
        {
            "person_id": [1],
            "condition_source_value": ["410"],
            "condition_start_date": pd.to_datetime(["2021-01-01"]),  # after index
        }
    )
    drug_df = pd.DataFrame(
        {
            "person_id": [1],
            "drug_concept_id": [101],
            "drug_exposure_start_date": pd.to_datetime(["2021-01-01"]),  # after index
        }
    )
    measurement_df = pd.DataFrame(
        {
            "person_id": [1],
            "measurement_concept_id": [3004249],
            "value_as_number": [200.0],
            "measurement_date": pd.to_datetime(["2021-01-01"]),  # after index
        }
    )
    index_dates = pd.Series({1: pd.Timestamp("2020-01-01")})
    result = make_feature_matrix(
        person_df,
        condition_df,
        drug_df,
        measurement_df,
        index_dates=index_dates,
        lookback_days=365,
    )
    row = result.iloc[0]
    assert not bool(row["myocardial_infarction"])
    assert row["drug_count"] == 0
    assert pd.isna(row.get("3004249_mean")) or row.get("3004249_mean") is None
