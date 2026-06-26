"""Criteria-driven OMOP cohort building utilities.

Operates on standardised OMOP column names. All functions are app-agnostic
(no ADE- or trial-specific logic lives here).
"""

from __future__ import annotations

import pandas as pd


def apply_inclusion_exclusion(
    df: pd.DataFrame,
    inclusion: dict,
    exclusion: dict,
) -> pd.DataFrame:
    """Filter *df* by column-value criteria dicts.

    Parameters
    ----------
    df:
        DataFrame to filter (typically a person-level table).
    inclusion:
        ``{column: value_or_list}`` — rows must satisfy ALL criteria to be kept.
        A list value means "column value must be in the list".
    exclusion:
        ``{column: value_or_list}`` — rows matching ANY criterion are dropped.

    Returns
    -------
    pd.DataFrame
        Filtered copy; original index preserved.
    """
    mask = pd.Series(True, index=df.index)

    for col, val in inclusion.items():
        if col not in df.columns:
            raise KeyError(f"Inclusion column '{col}' not found in DataFrame.")
        if isinstance(val, (list, set, tuple)):
            mask &= df[col].isin(val)
        else:
            mask &= df[col] == val

    result = df[mask].copy()

    for col, val in exclusion.items():
        if col not in result.columns:
            raise KeyError(f"Exclusion column '{col}' not found in DataFrame.")
        if isinstance(val, (list, set, tuple)):
            excl_mask = result[col].isin(val)
        else:
            excl_mask = result[col] == val
        result = result[~excl_mask]

    return result


def build_cohort(
    person_df: pd.DataFrame,
    inclusion: dict,
    exclusion: dict,
) -> pd.DataFrame:
    """Build a cohort by applying inclusion and exclusion criteria to *person_df*.

    Parameters
    ----------
    person_df:
        OMOP ``person`` table (or a pre-joined person-level DataFrame).
    inclusion:
        ``{column: value_or_list}`` — rows must satisfy ALL criteria to be kept.
    exclusion:
        ``{column: value_or_list}`` — rows matching ANY criterion are dropped.

    Returns
    -------
    pd.DataFrame
        Cohort DataFrame with the same columns as *person_df*.

    Example
    -------
    >>> cohort = build_cohort(
    ...     person_df,
    ...     inclusion={"gender_concept_id": 8507, "year_of_birth": list(range(1950, 2000))},
    ...     exclusion={"race_concept_id": 0},
    ... )
    """
    return apply_inclusion_exclusion(person_df, inclusion, exclusion)


def add_index_date(
    df: pd.DataFrame,
    index_date_col: str,
) -> pd.DataFrame:
    """Validate and set a cohort index-date column, coercing it to datetime.

    Parameters
    ----------
    df:
        Cohort DataFrame that must already contain *index_date_col*.
    index_date_col:
        Name of the column holding index dates (e.g. ``"cohort_start_date"``).

    Returns
    -------
    pd.DataFrame
        Copy of *df* with *index_date_col* cast to ``datetime64[ns]`` and any
        rows with a null index date dropped.

    Raises
    ------
    KeyError
        If *index_date_col* is not present in *df*.
    ValueError
        If the column cannot be parsed as dates.
    """
    if index_date_col not in df.columns:
        raise KeyError(f"Column '{index_date_col}' not found in DataFrame.")

    result = df.copy()
    result[index_date_col] = pd.to_datetime(result[index_date_col], errors="raise")
    null_count = result[index_date_col].isna().sum()
    if null_count:
        result = result.dropna(subset=[index_date_col])

    return result
