"""OMOP feature engineering utilities.

Every time-windowed function accepts ``index_dates`` and ``lookback_days`` so
callers can enforce temporal validity — only data recorded BEFORE a prediction
timepoint is included.  Pass ``index_dates=None`` to skip windowing entirely.
"""

from __future__ import annotations

import pandas as pd

from omop_mlops.constants import CHARLSON_ICD9_GROUPS


def normalize_icd(code: object) -> str:
    """Normalize an ICD-9/10 code for prefix matching.

    Removes dots and surrounding whitespace and upper-cases the result, so
    ``"250.0"``, ``"250.00"`` and MIMIC's ``"25000"`` all compare on the same
    footing.
    """
    return str(code).strip().replace(".", "").upper()


def _pre_index_mask(
    df: pd.DataFrame,
    date_col: str,
    index_dates: pd.Series,
    lookback_days: int | None,
) -> pd.Series:
    """Boolean mask: rows dated strictly BEFORE each person's index date.

    When *lookback_days* is given, rows older than that window are also
    excluded.  Persons missing from *index_dates* get no rows (NaT compares
    False), which avoids silently admitting undated history.
    """
    dates = pd.to_datetime(df[date_col])
    index_for_row = df["person_id"].map(index_dates)
    mask = dates < index_for_row
    if lookback_days is not None:
        cutoff = index_for_row - pd.to_timedelta(lookback_days, unit="D")
        mask &= dates >= cutoff
    return mask.fillna(False).astype(bool)


def comorbidity_flags(
    condition_df: pd.DataFrame,
    person_ids: list[int] | pd.Index,
    mapping: dict[str, list[str]] | None = None,
    index_dates: pd.Series | None = None,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Compute Boolean comorbidity flags per person from condition_occurrence rows.

    Parameters
    ----------
    condition_df:
        OMOP ``condition_occurrence`` table.  Must contain ``person_id`` and
        ``condition_source_value`` (ICD-9/10 string).  When ``index_dates`` is
        given it must also contain ``condition_start_date``.
    person_ids:
        Universe of person IDs to include (rows with no matching conditions get
        all-False flags).
    mapping:
        ``{comorbidity_name: [icd_prefix, ...]}``.  Defaults to
        :data:`~omop_mlops.constants.CHARLSON_ICD9_GROUPS`.  Prefixes are
        matched after normalization (dots removed), so ``"250.0"`` matches
        MIMIC's ``"25000"``.
    index_dates:
        ``pd.Series`` indexed by ``person_id`` with the prediction timepoint.
        Only conditions starting strictly before it are counted.
    lookback_days:
        Restrict to conditions within this many days before ``index_dates``.
        Ignored when ``index_dates`` is ``None``.

    Returns
    -------
    pd.DataFrame
        One row per ``person_id``, one Boolean column per comorbidity.
    """
    if mapping is None:
        mapping = CHARLSON_ICD9_GROUPS

    person_ids = pd.Index(person_ids).unique()
    df = condition_df[condition_df["person_id"].isin(person_ids)].copy()

    if index_dates is not None:
        if "condition_start_date" not in df.columns:
            raise KeyError(
                "condition_df must contain 'condition_start_date' when index_dates is given."
            )
        df = df[_pre_index_mask(df, "condition_start_date", index_dates, lookback_days)]

    codes = df["condition_source_value"].map(normalize_icd).astype(str)

    result = pd.DataFrame({"person_id": person_ids}).set_index("person_id")
    for label, prefixes in mapping.items():
        match = pd.Series(False, index=df.index)
        for prefix in prefixes:
            match |= codes.str.startswith(normalize_icd(prefix))
        has_condition = df.loc[match.astype(bool), "person_id"].drop_duplicates()
        result[label] = result.index.isin(has_condition)

    return result.reset_index()


def medication_counts(
    drug_df: pd.DataFrame,
    person_ids: list[int] | pd.Index,
    lookback_days: int | None = None,
    index_dates: pd.Series | None = None,
) -> pd.DataFrame:
    """Count distinct drugs per person, optionally within a lookback window.

    Only records where ``drug_exposure_start_date < index_date`` AND within
    *lookback_days* before the index date are counted when both
    ``index_dates`` and ``lookback_days`` are provided.

    Parameters
    ----------
    drug_df:
        OMOP ``drug_exposure`` table.  Must contain ``person_id``,
        ``drug_concept_id``, and ``drug_exposure_start_date``.
    person_ids:
        Universe of person IDs.
    lookback_days:
        Restrict to records within this many days before ``index_date``.
        Ignored when ``index_dates`` is ``None``.
    index_dates:
        ``pd.Series`` indexed by ``person_id`` holding the prediction timepoint
        for each person.  Records ON or AFTER this date are excluded.

    Returns
    -------
    pd.DataFrame
        Columns: ``person_id``, ``drug_count`` (int).
    """
    person_ids = pd.Index(person_ids).unique()
    df = drug_df[drug_df["person_id"].isin(person_ids)].copy()

    if index_dates is not None:
        df["drug_exposure_start_date"] = pd.to_datetime(df["drug_exposure_start_date"])
        df = df[_pre_index_mask(df, "drug_exposure_start_date", index_dates, lookback_days)]

    counts = (
        df.groupby("person_id")["drug_concept_id"]
        .nunique()
        .rename("drug_count")
        .reindex(person_ids, fill_value=0)
        .rename_axis("person_id")
        .reset_index()
    )
    return counts


def aggregate_measurements(
    measurement_df: pd.DataFrame,
    person_ids: list[int] | pd.Index,
    agg: tuple[str, ...] = ("mean", "min", "max"),
    index_dates: pd.Series | None = None,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Pivot and aggregate numeric measurements per person.

    Parameters
    ----------
    measurement_df:
        OMOP ``measurement`` table.  Must contain ``person_id``,
        ``measurement_concept_id``, and ``value_as_number``.  When
        ``index_dates`` is given it must also contain ``measurement_date``.
    person_ids:
        Universe of person IDs.
    agg:
        Aggregation functions to apply (any pandas groupby agg names).
    index_dates:
        ``pd.Series`` indexed by ``person_id`` with the prediction timepoint.
        Only measurements taken strictly before it are aggregated.
    lookback_days:
        Restrict to measurements within this many days before ``index_dates``.
        Ignored when ``index_dates`` is ``None``.

    Returns
    -------
    pd.DataFrame
        One row per ``person_id``; columns are
        ``{concept_id}_{agg}`` (e.g. ``3004249_mean``).
    """
    person_ids = pd.Index(person_ids).unique()
    df = measurement_df[measurement_df["person_id"].isin(person_ids)].copy()

    if index_dates is not None:
        if "measurement_date" not in df.columns:
            raise KeyError(
                "measurement_df must contain 'measurement_date' when index_dates is given."
            )
        df = df[_pre_index_mask(df, "measurement_date", index_dates, lookback_days)]

    df["value_as_number"] = pd.to_numeric(df["value_as_number"], errors="coerce")
    df = df.dropna(subset=["value_as_number"])

    if df.empty:
        return pd.DataFrame({"person_id": person_ids})

    pivot = (
        df.groupby(["person_id", "measurement_concept_id"])["value_as_number"]
        .agg(list(agg))
        .unstack("measurement_concept_id")
    )
    pivot.columns = [f"{concept}_{fn}" for fn, concept in pivot.columns]
    pivot = pivot.reindex(person_ids)
    return pivot.reset_index().rename(columns={"index": "person_id"})


def make_feature_matrix(
    person_df: pd.DataFrame,
    condition_df: pd.DataFrame,
    drug_df: pd.DataFrame,
    measurement_df: pd.DataFrame | None = None,
    index_dates: pd.Series | None = None,
    lookback_days: int | None = None,
) -> pd.DataFrame:
    """Join comorbidities, medication counts, and measurements into one feature matrix.

    Temporal validity: when ``index_dates`` is provided, every sub-function
    receives it so only data recorded BEFORE the prediction timepoint is used.
    This prevents label leakage.

    Parameters
    ----------
    person_df:
        OMOP ``person`` table; must contain ``person_id``.
    condition_df:
        OMOP ``condition_occurrence`` table.
    drug_df:
        OMOP ``drug_exposure`` table.
    measurement_df:
        OMOP ``measurement`` table.  Omit to skip measurement features.
    index_dates:
        ``pd.Series`` indexed by ``person_id``; the prediction timepoint per
        person.  Pass ``None`` to use all historical data.
    lookback_days:
        Days before ``index_date`` to look back.  Requires ``index_dates``.

    Returns
    -------
    pd.DataFrame
        One row per ``person_id`` from *person_df*.
    """
    person_ids = person_df["person_id"].unique()

    comorbidity_df = comorbidity_flags(
        condition_df, person_ids, index_dates=index_dates, lookback_days=lookback_days
    )
    drug_count_df = medication_counts(
        drug_df, person_ids, lookback_days=lookback_days, index_dates=index_dates
    )

    result = comorbidity_df.merge(drug_count_df, on="person_id", how="outer")

    if measurement_df is not None and not measurement_df.empty:
        meas_df = aggregate_measurements(
            measurement_df,
            person_ids,
            index_dates=index_dates,
            lookback_days=lookback_days,
        )
        result = result.merge(meas_df, on="person_id", how="left")

    # Ensure every person_id from person_df appears exactly once
    result = pd.DataFrame({"person_id": person_ids}).merge(result, on="person_id", how="left")

    bool_cols = comorbidity_df.columns.drop("person_id").tolist()
    for col in bool_cols:
        if col in result.columns:
            result[col] = result[col].fillna(False).astype(bool)

    if "drug_count" in result.columns:
        result["drug_count"] = result["drug_count"].fillna(0).astype(int)

    return result.reset_index(drop=True)
