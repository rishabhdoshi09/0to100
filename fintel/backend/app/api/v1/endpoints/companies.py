from __future__ import annotations
from fastapi import APIRouter
from app.api.deps import CurrentUser, DBSession
from app.core.exceptions import NotFoundError
from app.models.company import Company
from app.repositories.base import BaseRepository
from app.schemas.common import PaginatedResponse, PaginationParams
from app.schemas.company import CompanyListItem, CompanyOut

router = APIRouter(prefix="/companies", tags=["companies"])


@router.get("", response_model=PaginatedResponse[CompanyListItem])
async def list_companies(session: DBSession, current_user: CurrentUser, params: PaginationParams = PaginationParams()):
    repo = BaseRepository(Company, session)
    companies, total = await repo.list(offset=params.offset, limit=params.limit)
    return PaginatedResponse(
        data=[CompanyListItem.model_validate(c) for c in companies],
        total=total,
        offset=params.offset,
        limit=params.limit,
    )


@router.get("/{symbol}", response_model=CompanyOut)
async def get_company(symbol: str, session: DBSession, current_user: CurrentUser):
    repo = BaseRepository(Company, session)
    company = await repo.get_by_field("symbol", symbol.upper())
    if not company:
        raise NotFoundError(f"Company '{symbol}' not found")
    return CompanyOut.model_validate(company)


@router.post("/{symbol}/refresh", status_code=202)
async def refresh_company(symbol: str, current_user: CurrentUser):
    from app.workers.tasks.scraping import scrape_company
    scrape_company.delay(symbol.upper())
    return {"message": f"Refresh queued for {symbol.upper()}"}


@router.get("/{symbol}/filings")
async def get_filings(symbol: str, session: DBSession, current_user: CurrentUser):
    from app.models.filing import Filing
    from sqlalchemy import select
    from app.models.company import Company as CompanyModel
    company_stmt = select(CompanyModel).where(CompanyModel.symbol == symbol.upper())
    company = (await session.execute(company_stmt)).scalar_one_or_none()
    if not company:
        raise NotFoundError(f"Company '{symbol}' not found")
    filing_stmt = select(Filing).where(Filing.company_id == company.id).order_by(Filing.created_at.desc()).limit(50)
    filings = list((await session.execute(filing_stmt)).scalars().all())
    return {"symbol": symbol.upper(), "filings": [{"id": str(f.id), "title": f.title, "type": f.filing_type, "date": f.filing_date, "status": f.processing_status} for f in filings]}
