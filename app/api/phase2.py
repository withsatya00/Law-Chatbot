from fastapi import APIRouter

from app.schemas.phase2 import (
    CyberFraudRequest,
    CyberFraudResponse,
    EvidenceOrganizeRequest,
    EvidenceOrganizeResponse,
    JurisdictionRequest,
    JurisdictionResponse,
    TimelineRequest,
    TimelineResponse,
)
from app.services.phase2_workflow import (
    analyze_timeline,
    cyber_fraud_workflow,
    jurisdiction_check,
    organize_evidence,
)

router = APIRouter(prefix="/workflows", tags=["phase-2-workflows"])


@router.post("/cyber-fraud", response_model=CyberFraudResponse)
async def cyber_fraud(request: CyberFraudRequest) -> CyberFraudResponse:
    return cyber_fraud_workflow(request)


@router.post("/evidence/organize", response_model=EvidenceOrganizeResponse)
async def evidence_organizer(request: EvidenceOrganizeRequest) -> EvidenceOrganizeResponse:
    return organize_evidence(request)


@router.post("/timeline", response_model=TimelineResponse)
async def timeline(request: TimelineRequest) -> TimelineResponse:
    return analyze_timeline(request)


@router.post("/jurisdiction", response_model=JurisdictionResponse)
async def jurisdiction(request: JurisdictionRequest) -> JurisdictionResponse:
    return jurisdiction_check(request)
