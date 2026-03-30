import json
import re
from typing import Optional

from openai import OpenAI

from .models import IntentDetectionResult, IntentEntities, QueryIntent


# Multi-word phrases first (checked before single-token matches) to avoid
# recommendation keywords winning on e.g. "top ways to reach Goa".
TRIP_PHRASES = (
    "how do i reach",
    "how to reach",
    "ways to reach",
    "how do i get",
    "how to get",
    "get there",
    "reach there",
    "directions to",
    "directions from",
    "route to",
    "plan a trip",
    "plan my trip",
    "plan our trip",
    "planning a trip",
    "planning my trip",
    "planning our trip",
    "planning to",
    "help me plan",
    "want to plan",
    "need to plan",
    "schedule a trip",
    "organize a trip",
    "book a trip",
    "trip around",
    "around it",
    "weekend plan",
    "day-wise",
    "day 1",
    "2 day",
    "3 day",
)
TRIP_KEYWORDS = (
    "itinerary",
    "days",
)
RECOMMENDATION_KEYWORDS = (
    "recommend",
    "suggest",
    "best",
    "top",
    "must try",
    "must-try",
    "good for",
    "options",
)


def _extract_destination(query: str) -> Optional[str]:
    """
    Best-effort destination extraction from common natural language forms.
    """
    patterns = [
        r"\bin\s+([A-Za-z][A-Za-z\s]{1,40})",
        r"\bfor\s+([A-Za-z][A-Za-z\s]{1,40})\s+trip\b",
        r"\bto\s+([A-Za-z][A-Za-z\s]{1,40})",
    ]
    for pattern in patterns:
        match = re.search(pattern, query, flags=re.IGNORECASE)
        if not match:
            continue
        value = (match.group(1) or "").strip(" .?!,")
        if value:
            return value.title()
    return None


def _has_trip_planning_signal(lowered: str) -> bool:
    if any(phrase in lowered for phrase in TRIP_PHRASES):
        return True
    if any(keyword in lowered for keyword in TRIP_KEYWORDS):
        return True
    # Word-boundary match: include "planning" (not matched by \bplan\b alone).
    if re.search(r"\b(plan|planning|trip|itinerary)\b", lowered):
        return True
    if re.search(r"\b\d+\s*day\b", lowered):
        return True
    return False


def _merge_conversation_for_intent(query: str, conversation_context: Optional[str]) -> str:
    q = (query or "").strip()
    ctx = (conversation_context or "").strip()
    if not ctx:
        return q
    return f"{ctx}\n{q}".strip()


def _rule_based_intent(query: str, conversation_context: Optional[str] = None) -> IntentDetectionResult:
    lowered = (query or "").strip().lower()
    destination = _extract_destination(lowered)
    if not destination and conversation_context:
        destination = _extract_destination(conversation_context.lower())
    entities = IntentEntities(destination=destination)

    if _has_trip_planning_signal(lowered):
        return IntentDetectionResult(
            intent=QueryIntent.trip_planning,
            confidence=0.92,
            entities=entities,
            reason="rule:trip_keyword",
        )
    if any(keyword in lowered for keyword in RECOMMENDATION_KEYWORDS):
        return IntentDetectionResult(
            intent=QueryIntent.recommendation,
            confidence=0.87,
            entities=entities,
            reason="rule:recommendation_keyword",
        )
    return IntentDetectionResult(
        intent=QueryIntent.search,
        confidence=0.75,
        entities=entities,
        reason="rule:default_search",
    )


def _safe_json_parse(content: str) -> dict:
    cleaned = (content or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)
    try:
        return json.loads(cleaned)
    except Exception:
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start >= 0 and end > start:
            try:
                return json.loads(cleaned[start : end + 1])
            except Exception:
                return {}
    return {}


def _llm_intent(
    query: str,
    client: OpenAI,
    conversation_context: Optional[str] = None,
) -> IntentDetectionResult:
    user_block = query.strip()
    if conversation_context:
        user_block = (
            "Classify intent using only the latest user message.\n"
            "Use conversation only to infer missing entities (for example, resolve 'there' to a prior destination).\n\n"
            "Conversation so far (oldest first):\n"
            f"{conversation_context.strip()}\n\n"
            f"Latest message:\n{query.strip()}"
        )
    response = client.chat.completions.create(
        model="gpt-4o-mini",
        temperature=0.0,
        response_format={"type": "json_object"},
        messages=[
            {
                "role": "system",
                "content": (
                    "Classify the user query intent for a travel reels assistant.\n"
                    "Return strict JSON with keys: intent, confidence, entities.\n"
                    "intent must be one of: trip_planning, search, recommendation, unknown.\n"
                    "confidence must be 0..1.\n"
                    "entities object keys: destination, origin, departure_date, return_date, dates, budget, trip_length, food_pref.\n"
                    "Classify intent from the latest message only; do not let earlier conversation force trip_planning.\n"
                    "Use trip_planning when the user asks how to get somewhere, directions, routes, transportation, "
                    "multi-day itineraries, planning travel around a place, or follow-ups like reaching a prior result.\n"
                    "Infer destination from earlier turns when the latest message is vague (e.g. 'plan a trip there').\n"
                    "Use recommendation for open-ended best/top/must-try lists of places without routing or day-by-day planning.\n"
                    "Use search for finding or listing saved reels by topic or location.\n"
                    "If latest message is only a location/topic name (e.g. 'Kasol', 'Igatpuri cafes') without route/planning terms, classify as search."
                ),
            },
            {"role": "user", "content": user_block},
        ],
    )
    content = response.choices[0].message.content or "{}"
    payload = _safe_json_parse(content)

    raw_intent = str(payload.get("intent") or "unknown").strip().lower()
    if raw_intent not in {i.value for i in QueryIntent}:
        raw_intent = QueryIntent.unknown.value

    try:
        confidence = float(payload.get("confidence", 0.0))
    except Exception:
        confidence = 0.0
    confidence = max(0.0, min(1.0, confidence))

    raw_entities = payload.get("entities") or {}
    if not isinstance(raw_entities, dict):
        raw_entities = {}
    entities = IntentEntities(
        destination=raw_entities.get("destination"),
        origin=raw_entities.get("origin"),
        departure_date=raw_entities.get("departure_date"),
        return_date=raw_entities.get("return_date"),
        dates=raw_entities.get("dates"),
        budget=raw_entities.get("budget"),
        trip_length=raw_entities.get("trip_length"),
        food_pref=raw_entities.get("food_pref"),
    )

    intent = QueryIntent(raw_intent)
    clarification_needed = confidence < 0.6 or intent == QueryIntent.unknown
    if clarification_needed and intent == QueryIntent.unknown:
        intent = QueryIntent.search

    return IntentDetectionResult(
        intent=intent,
        confidence=confidence,
        entities=entities,
        clarification_needed=clarification_needed,
        reason="llm_classifier",
    )


def detect_intent(
    query: str,
    client: OpenAI,
    conversation_context: Optional[str] = None,
) -> IntentDetectionResult:
    """
    Rule-first intent detection with LLM fallback for ambiguous cases.
    """
    rule_result = _rule_based_intent(query, conversation_context=conversation_context)

    # Rules are enough for strongly-signaled queries.
    if rule_result.confidence >= 0.86:
        return rule_result

    try:
        llm_result = _llm_intent(query, client, conversation_context=conversation_context)
    except Exception:
        return rule_result

    # Default search (0.75) often loses to a correct but modest-confidence LLM trip
    # because 0.62 < 0.75. Prefer trip_planning when the model is reasonably sure.
    if (
        llm_result.intent == QueryIntent.trip_planning
        and llm_result.confidence >= 0.52
        and rule_result.intent == QueryIntent.search
    ):
        return llm_result

    if llm_result.confidence >= rule_result.confidence:
        return llm_result

    return rule_result
