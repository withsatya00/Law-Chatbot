import asyncio

from app.recommendation.engine import LawyerRecommendationEngine


def test_lawyer_recommendation_for_family_law() -> None:
    recommendation = asyncio.run(LawyerRecommendationEngine().recommend("Divorce", "Family Law"))
    assert recommendation.category == "Family Lawyer"
    assert recommendation.confidence > 0.8
