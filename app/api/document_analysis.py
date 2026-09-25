from fastapi import APIRouter, Depends

from app.api.deps import get_current_user_id
from app.schemas.document import DocumentAnalysisRequest, DocumentAnalysisResponse
from app.services.document_service import DocumentService

router = APIRouter(tags=["document-analysis"])


@router.post("/document-analysis", response_model=DocumentAnalysisResponse)
async def analyze_document(
    request: DocumentAnalysisRequest, user_id: str | None = Depends(get_current_user_id),
) -> DocumentAnalysisResponse:
    return await DocumentService().analyze(request, authenticated_user_id=user_id)
