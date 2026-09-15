"""Knowledge base concepts API endpoints."""

from typing import List, Optional
from fastapi import APIRouter, Depends, HTTPException, Query, status

from app.schemas.concept import ConceptDetailResponse, ConceptSummaryResponse
from app.services.ml_service import MLService, get_ml_service

router = APIRouter(prefix="/concepts", tags=["Reflective Concepts"])


@router.get(
    "",
    response_model=List[ConceptSummaryResponse],
    summary="List curated psychological and philosophical concepts",
)
def list_concepts(
    category: Optional[str] = Query(None, description="Filter by category (e.g. philosophical, cognitive_reflective)"),
    search: Optional[str] = Query(None, description="Search term for concept name, definition, or pattern"),
    ml_service: MLService = Depends(get_ml_service),
) -> List[ConceptSummaryResponse]:
    """Browse the curated 20-concept development knowledge base."""
    concepts = ml_service.list_concepts(category=category, search=search)
    return [
        ConceptSummaryResponse(
            id=c["id"],
            name=c["name"],
            category=c["category"],
            definition=c["definition"],
        )
        for c in concepts
    ]


@router.get(
    "/{concept_id}",
    response_model=ConceptDetailResponse,
    summary="Get full details, source citation, and cautions for a concept",
)
def get_concept(
    concept_id: str,
    ml_service: MLService = Depends(get_ml_service),
) -> ConceptDetailResponse:
    """Retrieve full concept documentation including definition, reflective lens, citation, and cautions."""
    concept = ml_service.get_concept(concept_id)
    if not concept:
        raise HTTPException(
            status_code=status.HTTP_404_NOT_FOUND,
            detail=f"Concept '{concept_id}' not found.",
        )
    return ConceptDetailResponse(**concept)
