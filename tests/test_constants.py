"""Tests for omop_mlops.constants."""

from omop_mlops.constants import (
    CHARLSON_ICD9_GROUPS,
    ICD9_DRUG_ADE_ECODE_RANGE,
    OMOP_DOMAINS,
    OmopTable,
)


def test_omop_table_enum_has_expected_members():
    expected = {
        "PERSON",
        "VISIT_OCCURRENCE",
        "CONDITION_OCCURRENCE",
        "DRUG_EXPOSURE",
        "MEASUREMENT",
        "NOTE",
    }
    assert set(OmopTable.__members__) == expected


def test_omop_table_values_are_strings():
    for member in OmopTable:
        assert isinstance(member.value, str)


def test_icd9_ade_ecode_range_is_tuple_of_two_strings():
    assert isinstance(ICD9_DRUG_ADE_ECODE_RANGE, tuple)
    assert len(ICD9_DRUG_ADE_ECODE_RANGE) == 2
    assert ICD9_DRUG_ADE_ECODE_RANGE[0] == "E930"
    assert ICD9_DRUG_ADE_ECODE_RANGE[1] == "E949"


def test_charlson_groups_is_dict_of_lists():
    assert isinstance(CHARLSON_ICD9_GROUPS, dict)
    assert len(CHARLSON_ICD9_GROUPS) >= 3
    for key, val in CHARLSON_ICD9_GROUPS.items():
        assert isinstance(key, str)
        assert isinstance(val, list)
        assert len(val) >= 1


def test_omop_domains_has_expected_keys():
    assert "condition" in OMOP_DOMAINS
    assert "drug" in OMOP_DOMAINS
    assert "measurement" in OMOP_DOMAINS
    for v in OMOP_DOMAINS.values():
        assert isinstance(v, int)
