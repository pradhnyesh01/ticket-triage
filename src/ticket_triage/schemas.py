"""Pydantic request/response schemas for the serving API.

PredictRequest accepting only `narrative` is itself part of the
leakage-boundary enforcement, not just documentation of one: it is
structurally impossible to construct a request carrying an outcome
column (company response, timeliness, tags), because the schema has no
field for one.
"""

from pydantic import BaseModel, Field


class PredictRequest(BaseModel):
    narrative: str = Field(
        ...,
        min_length=1,
        description="The consumer complaint narrative text -- the only input either model ever sees.",
    )


class PredictResponse(BaseModel):
    product: str = Field(..., description="Predicted Product category (DistilBERT, fine-tuned)")
    product_confidence: float = Field(..., ge=0, le=1)
    priority: str = Field(
        ...,
        description="Predicted Priority: Low or Elevated (binary Logistic Regression -- "
        "see README.md for why the 3-way Low/Medium/High models aren't served)",
    )
    priority_confidence: float = Field(..., ge=0, le=1)
