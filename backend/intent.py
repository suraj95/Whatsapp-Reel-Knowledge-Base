import json
import re
from typing import Optional

from openai import OpenAI

from .models import IntentDetectionResult, IntentEntities, QueryIntent


TRIP_KEYWORDS = (
    "plan",
    "itinerary",
    "day 1",
    "day-wise",
    "days",
    "trip",
    "2 day",
    "3 day",
    "weekend plan",
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


def _rule_based_intent(query: str) -> IntentDetectionResult:
    lowered = (query or "").strip().lower()
    destination = _extract_destination(lowered)
    entities = IntentEntities(destination=destination)

    if any(keyword in lowered for keyword in TRIP_KEYWORDS):
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


def _llm_intent(query: str, client: OpenAI) -> IntentDetectionResult:
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
                    "entities object keys: destination, dates, budget, trip_length, food_pref."
                ),
            },
            {"role": "user", "content": query},
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


def detect_intent(query: str, client: OpenAI) -> IntentDetectionResult:
    """
    Rule-first intent detection with LLM fallback for ambiguous cases.
    """
    rule_result = _rule_based_intent(query)

    # Rules are enough for strongly-signaled queries.
    if rule_result.confidence >= 0.86:
        return rule_result

    try:
        llm_result = _llm_intent(query, client)
        if llm_result.confidence >= rule_result.confidence:
            return llm_result
    except Exception:
        # Fallback to deterministic behavior on any model/API failure.
        pass

    return rule_result
