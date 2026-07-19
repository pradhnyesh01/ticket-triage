"""Feature extraction for both prediction heads.

Both heads (Product category, Priority bucket) are predicted from the
consumer complaint narrative alone -- see CLAUDE.md and
Project1_Build_Plan.docx. No other column, especially not the outcome
fields used to build the priority label (company_response_to_consumer,
timely_response, tags, company_public_response), may reach a model as
input, at training or inference time.
"""

from typing import Mapping, TypedDict


class ModelFeatures(TypedDict):
    narrative: str


# Columns that exist in product_stratified_sample but must never be used
# as model input: the raw material behind the priority label. Kept as an
# explicit, named set so the invariant reads the same in code as it does
# in the build plan and README, not just implied by the function body.
LEAKAGE_COLUMNS = frozenset(
    {
        "company_response_to_consumer",
        "timely_response",
        "tags",
        "company_public_response",
    }
)


def extract_model_features(row: Mapping[str, object]) -> ModelFeatures:
    """Extract the only field a model may ever see: narrative text."""
    return {"narrative": row["consumer_complaint_narrative"]}
