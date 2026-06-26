"""Tests for omop_mlops.cohort."""

import pandas as pd
import pytest

from omop_mlops.cohort import add_index_date, build_cohort


@pytest.fixture()
def person_df():
    return pd.DataFrame(
        {
            "person_id": [1, 2, 3, 4, 5],
            "gender_concept_id": [8507, 8532, 8507, 8532, 8507],
            "year_of_birth": [1960, 1975, 1990, 1955, 1980],
            "race_concept_id": [8527, 8527, 0, 8516, 8527],
        }
    )


def test_build_cohort_inclusion_single_value(person_df):
    result = build_cohort(person_df, inclusion={"gender_concept_id": 8507}, exclusion={})
    assert set(result["person_id"]) == {1, 3, 5}


def test_build_cohort_inclusion_list_value(person_df):
    result = build_cohort(person_df, inclusion={"year_of_birth": [1960, 1975]}, exclusion={})
    assert set(result["person_id"]) == {1, 2}


def test_build_cohort_exclusion_removes_rows(person_df):
    result = build_cohort(person_df, inclusion={}, exclusion={"race_concept_id": 0})
    assert 3 not in result["person_id"].values
    assert len(result) == 4


def test_build_cohort_combined_criteria(person_df):
    result = build_cohort(
        person_df,
        inclusion={"gender_concept_id": 8507},
        exclusion={"year_of_birth": [1990]},
    )
    assert set(result["person_id"]) == {1, 5}


def test_build_cohort_returns_dataframe_with_same_columns(person_df):
    result = build_cohort(person_df, inclusion={}, exclusion={})
    assert list(result.columns) == list(person_df.columns)


def test_add_index_date_coerces_strings():
    df = pd.DataFrame({"person_id": [1, 2], "cohort_start_date": ["2020-01-01", "2021-06-15"]})
    result = add_index_date(df, "cohort_start_date")
    assert pd.api.types.is_datetime64_any_dtype(result["cohort_start_date"])


def test_add_index_date_drops_nulls():
    df = pd.DataFrame(
        {"person_id": [1, 2, 3], "cohort_start_date": ["2020-01-01", None, "2021-06-15"]}
    )
    result = add_index_date(df, "cohort_start_date")
    assert len(result) == 2
    assert result["person_id"].tolist() == [1, 3]


def test_add_index_date_raises_on_missing_column():
    df = pd.DataFrame({"person_id": [1]})
    with pytest.raises(KeyError):
        add_index_date(df, "cohort_start_date")
