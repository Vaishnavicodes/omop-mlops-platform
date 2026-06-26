"""OMOP-CDM domain constants and concept-ID reference maps."""

from enum import Enum

# ---------------------------------------------------------------------------
# OMOP-CDM canonical table names
# ---------------------------------------------------------------------------


class OmopTable(str, Enum):
    PERSON = "person"
    VISIT_OCCURRENCE = "visit_occurrence"
    CONDITION_OCCURRENCE = "condition_occurrence"
    DRUG_EXPOSURE = "drug_exposure"
    MEASUREMENT = "measurement"
    NOTE = "note"


# ---------------------------------------------------------------------------
# ICD-9 drug-induced adverse-event E-codes (E930–E949)
# ---------------------------------------------------------------------------

ICD9_DRUG_ADE_ECODE_RANGE = ("E930", "E949")

# ---------------------------------------------------------------------------
# Charlson Comorbidity Index — ICD-9 groupings
# Expand this dict with the full Deyo/Quan mappings before production use.
# ---------------------------------------------------------------------------

CHARLSON_ICD9_GROUPS: dict[str, list[str]] = {
    "myocardial_infarction": ["410", "412"],
    "congestive_heart_failure": ["428"],
    "diabetes_uncomplicated": ["250.0", "250.1", "250.2", "250.3"],
    "chronic_pulmonary_disease": ["490", "491", "492", "493", "494", "495", "496"],
    # ... add remaining Charlson categories (renal disease, liver disease, etc.)
}

# ---------------------------------------------------------------------------
# OMOP standard concept domains used for feature engineering
# ---------------------------------------------------------------------------

OMOP_DOMAINS = {
    "condition": 440,
    "drug": 13,
    "measurement": 21,
    "procedure": 10,
    "observation": 27,
}
