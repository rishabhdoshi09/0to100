from __future__ import annotations
import uuid
from pydantic import BaseModel


class CompanyListItem(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    symbol: str
    name: str
    exchange: str
    sector: str | None
    market_cap: float | None


class CompanyOut(BaseModel):
    model_config = {"from_attributes": True}

    id: uuid.UUID
    symbol: str
    name: str
    isin: str | None
    exchange: str
    sector: str | None
    industry: str | None
    market_cap: float | None
    description: str | None
    website: str | None
