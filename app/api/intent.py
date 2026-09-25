from fastapi import APIRouter

from app.entity_extraction.extractor import EntityExtractor
from app.intent.detector import IntentDetector
from app.schemas.analysis import EntityRequest, EntityResponse, IntentRequest, IntentResponse

router = APIRouter(tags=["analysis"])


@router.post("/intent", response_model=IntentResponse)
async def detect_intent(request: IntentRequest) -> IntentResponse:
    return await IntentDetector().detect(request.text, request.language)


@router.post("/entities", response_model=EntityResponse)
async def extract_entities(request: EntityRequest) -> EntityResponse:
    """Entities, legal roles and a timeline, with the evidence for each.

    The flat `entities` map is unchanged from before Phase 3; `structured`,
    `unresolved_roles`, `conflicts`, `timeline` and `questions` are additive.
    No model call is made from this route -- it is the deterministic layer
    only, so the response is reproducible and free.
    """
    return await EntityExtractor().extract_structured(request.text, request.language)
