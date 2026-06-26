"""Tests for omop_mlops.features."""

import pandas as pd
import pytest

from omop_mlops.features import (
    comorbidity_flags,
    make_feature_matrix,
    medication_counts,
)

# ---------------------------------------------------------------------------
# Synthetic OMOP fixtures
# ---------------------------------------------------------------------------


@pytest.fixture()
def person_df():
    return pd.DataFrame({"person_id": [1, 2, 3]})


@pytest.fixture()
def condition_df():
    return pd.DataFrame(
        {
            "person_id": [1, 1, 2, 3],
            "condition_source_value": ["410", "428", "250.0", "999"],
        }
    )


@pytest.fixture()
def drug_df():
    return pd.DataFrame(
        {
            "person_id": [1, 1, 2, 2, 3],
            "drug_concept_id": [101, 102, 101, 103, 104],
            "drug_exposure_start_date": pd.to_datetime(
                ["2020-01-10", "2020-03-01", "2020-02-15", "2020-04-01", "2020-05-01"]
            ),
        }
    )


@pytest.fixture()
def measurement_df():
    return pd.DataFrame(
        {
            "person_id": [1, 1, 2, 3],
            "measurement_concept_id": [3004249, 3004249, 3004249, 3013682],
            "value_as_number": [120.0, 130.0, 115.0, 75.0],
        }
    )


# ---------------------------------------------------------------------------
# comorbidity_flags
# ---------------------------------------------------------------------------


def test_comorbidity_flags_returns_one_row_per_person(condition_df, person_df):
    result = comorbidity_flags(condition_df, person_df["person_id"])
    assert len(result) == 3
    assert "person_id" in result.columns


def test_comorbidity_flags_correct_values(condition_df, person_df):
    result = comorbidity_flags(condition_df, person_df["person_id"])
    row1 = result[result["person_id"] == 1].iloc[0]
    assert bool(row1["myocardial_infarction"])
    assert bool(row1["congestive_heart_failure"])

    row3 = result[result["person_id"] == 3].iloc[0]
    assert not bool(row3["myocardial_infarction"])


# ---------------------------------------------------------------------------
# medication_counts
# ---------------------------------------------------------------------------


def test_medication_counts_basic(drug_df, person_df):
    result = medication_counts(drug_df, person_df["person_id"])
    assert len(result) == 3
    row1 = result[result["person_id"] == 1].iloc[0]
    assert row1["drug_count"] == 2  # two distinct drug_concept_ids


def test_medication_counts_lookback_window(drug_df, person_df):
    index_dates = pd.Series(
        {
            "1": pd.Timestamp("2020-02-01"),
            "2": pd.Timestamp("2020-03-01"),
            "3": pd.Timestamp("2020-06-01"),
        }
    )
    index_dates.index = [1, 2, 3]

    # lookback 30 days: person 1 only has drug on 2020-01-10 (within 30 days before 2020-02-01)
    result = medication_counts(
        drug_df, person_df["person_id"], lookback_days=30, index_dates=index_dates
    )
    row1 = result[result["person_id"] == 1].iloc[0]
    assert row1["drug_count"] == 1  # only 2020-01-10 is within 30 days before 2020-02-01


def test_medication_counts_excludes_on_or_after_index_date(drug_df, person_df):
    index_dates = pd.Series(
        {
            1: pd.Timestamp("2020-01-10"),
            2: pd.Timestamp("2020-01-01"),
            3: pd.Timestamp("2019-01-01"),
        }
    )
    result = medication_counts(drug_df, person_df["person_id"], index_dates=index_dates)
    row1 = result[result["person_id"] == 1].iloc[0]
    assert row1["drug_count"] == 0  # index date == first drug date, strict < excludes it


# ---------------------------------------------------------------------------
# make_feature_matrix
# ---------------------------------------------------------------------------


def test_make_feature_matrix_one_row_per_person(person_df, condition_df, drug_df, measurement_df):
    result = make_feature_matrix(person_df, condition_df, drug_df, measurement_df)
    assert len(result) == len(person_df)
    assert result["person_id"].nunique() == len(person_df)


def test_make_feature_matrix_includes_drug_count(person_df, condition_df, drug_df):
    result = make_feature_matrix(person_df, condition_df, drug_df)
    assert "drug_count" in result.columns


def test_make_feature_matrix_with_lookback(person_df, condition_df, drug_df):
    index_dates = pd.Series(
        {
            1: pd.Timestamp("2020-06-01"),
            2: pd.Timestamp("2020-06-01"),
            3: pd.Timestamp("2020-06-01"),
        }
    )
    result = make_feature_matrix(
        person_df, condition_df, drug_df, lookback_days=365, index_dates=index_dates
    )
    assert len(result) == 3
