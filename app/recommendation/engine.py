from typing import ClassVar

import structlog

from app.recommendation.directory import LawyerDirectory, get_directory
from app.schemas.common import LawyerRecommendation

log = structlog.get_logger(__name__)


def render_recommendation(recommendation: LawyerRecommendation, language: str) -> str:
    """Render actionable guidance without asking a model to invent listings."""
    documents = "\n".join(f"- {item}" for item in recommendation.documents_to_carry)
    questions = "\n".join(f"- {item}" for item in recommendation.questions_to_ask)
    listings = ""
    if recommendation.verified_listings:
        rows = "\n".join(
            f"- **{item.full_name}** — {item.city or item.state or 'location not supplied'}"
            for item in recommendation.verified_listings
        )
        listings = f"\n\n**Verified-directory results**\n{rows}"

    if language == "hindi":
        return (
            f"आपकी बताई हुई स्थिति के आधार पर **{recommendation.category}** से सलाह लेना उचित होगा। "
            "यह विशेषज्ञता आपकी बताई हुई कानूनी श्रेणी के आधार पर सुझाई गई है।\n\n"
            f"**प्राथमिकता:** {recommendation.urgency}\n"
            f"**क्षेत्राधिकार:** {recommendation.jurisdiction_note}\n\n"
            f"**साथ ले जाने वाले दस्तावेज़**\n{documents}\n\n"
            f"**वकील से पूछने योग्य सवाल**\n{questions}\n\n"
            f"_{recommendation.directory_notice}_{listings}"
        )
    if language == "hinglish":
        return (
            f"Aapki batayi hui situation ke liye **{recommendation.category}** se consult karna useful hoga. "
            "Ye specialization aapki legal category ke basis par suggest ki gayi hai.\n\n"
            f"**Priority:** {recommendation.urgency}\n"
            f"**Jurisdiction:** {recommendation.jurisdiction_note}\n\n"
            f"**Saath le jaane wale documents**\n{documents}\n\n"
            f"**Lawyer se poochhne wale sawal**\n{questions}\n\n"
            f"_{recommendation.directory_notice}_{listings}"
        )
    return (
        f"Based on what you have shared, consult a **{recommendation.category}**. "
        f"{recommendation.reason}\n\n"
        f"**Priority:** {recommendation.urgency}\n"
        f"**Jurisdiction:** {recommendation.jurisdiction_note}\n\n"
        f"**Documents to take**\n{documents}\n\n"
        f"**Questions to ask**\n{questions}\n\n"
        f"_{recommendation.directory_notice}_{listings}"
    )


class LawyerRecommendationEngine:
    mapping: ClassVar[dict[str, str]] = {
        "Labour Law": "Labour Lawyer",
        "Cyber Law": "Cyber Lawyer",
        "Banking and Criminal Law": "Cheque Bounce Lawyer",
        "Family Law": "Family Lawyer",
        "Property Law": "Property Lawyer",
        "Consumer Law": "Consumer Lawyer",
        "Criminal Law": "Criminal Lawyer",
        "Tax Law": "Tax Lawyer",
        "Employment Law": "Employment Lawyer",
        "Public Law": "RTI Lawyer",
        "General Law": "General Practice Lawyer",
    }

    documents: ClassVar[dict[str, list[str]]] = {
        "Family Law": ["identity proof", "marriage/relationship records", "relevant orders and communications"],
        "Property Law": ["title and registration documents", "tax/possession records", "agreement and notices"],
        "Cyber Law": ["transaction records", "screenshots and messages", "complaint/reference numbers"],
        "Banking and Criminal Law": ["original cheque and return memo", "demand notice and delivery proof", "transaction records"],
        "Criminal Law": ["FIR/complaint copy", "notices or court orders", "chronology and available evidence"],
        "Consumer Law": ["invoice or agreement", "complaints and replies", "payment and warranty records"],
        "Employment Law": ["appointment letter and policies", "salary records", "notices and communications"],
        "Labour Law": ["employment records", "wage records", "termination or dispute communications"],
        "Tax Law": ["notices and filed returns", "assessment/orders", "supporting accounts and payment records"],
        "Public Law": ["application and acknowledgement", "authority replies", "appeal or order records"],
    }

    urgent_terms: ClassVar[tuple[str, ...]] = (
        "arrest", "custody", "detained", "violence", "threat", "deadline", "today",
        "fraud", "stolen", "summons", "warrant", "eviction", "terminate",
    )

    def __init__(self, directory: LawyerDirectory | None = None) -> None:
        self.directory = directory or get_directory()

    async def recommend(
        self, intent: str, legal_category: str, *, city: str = "", language: str = ""
    ) -> LawyerRecommendation:
        category = self.mapping.get(legal_category, "General Practice Lawyer")
        confidence = 0.86 if legal_category in self.mapping else 0.62
        urgency = "prompt" if any(term in intent.lower() for term in self.urgent_terms) else "routine"
        listings = []
        directory_available = self.directory.available()
        if directory_available:
            try:
                listings = await self.directory.search(
                    specialization=category, city=city, language=language, limit=5
                )
            except (ConnectionError, TimeoutError, OSError) as exc:
                log.warning("lawyer_directory_unavailable", error=str(exc))
                directory_available = False
        documents = self.documents.get(
            legal_category,
            ["a written chronology", "all relevant notices or orders", "supporting documents and communications"],
        )
        return LawyerRecommendation(
            category=category,
            confidence=confidence,
            reason=f"The detected issue is {intent}, which falls under {legal_category}.",
            urgency=urgency,
            jurisdiction_note=(
                "Confirm the correct court/forum and engage an advocate entitled to practise there; "
                "venue can depend on the facts."
            ),
            documents_to_carry=documents,
            questions_to_ask=[
                "What limitation period or immediate deadline applies?",
                "Which court, tribunal, authority, or police station has jurisdiction?",
                "What evidence should I preserve before taking the next step?",
            ],
            directory_available=directory_available,
            directory_notice=(
                "Listings below came from the configured directory. Check enrolment and current standing independently."
                if listings
                else (
                    "A verified directory is connected, but it returned no matching listing. "
                    "This is a specialization suggestion, not a lawyer listing."
                    if directory_available
                    else "No verified lawyer directory is connected. This is a specialization suggestion, not a lawyer listing."
                )
            ),
            verified_listings=listings,
        )
