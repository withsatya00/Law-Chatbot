from fastapi import APIRouter
from pydantic import BaseModel

from app.recommendation.engine import LawyerRecommendationEngine
from app.schemas.common import LawyerRecommendation

router = APIRouter(tags=["recommendation"])


class RecommendRequest(BaseModel):
    intent: str
    legal_category: str
    city: str = ""
    language: str = ""


@router.post("/recommend-lawyer", response_model=LawyerRecommendation)
async def recommend_lawyer(request: RecommendRequest) -> LawyerRecommendation:
    return await LawyerRecommendationEngine().recommend(
        request.intent, request.legal_category, city=request.city, language=request.language
    )
