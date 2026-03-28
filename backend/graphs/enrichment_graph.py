"""LangGraph ReAct enrichment agent (replaces LangChain create_agent)."""

from __future__ import annotations

import os
from functools import lru_cache
from typing import Any, Dict, Optional

from langchain_core.messages import AIMessage, HumanMessage
from langchain_core.tools import tool
from langgraph.prebuilt import create_react_agent


ENRICHMENT_SYSTEM = """You are an enrichment agent for a travel/food reel database.
Given a raw vision summary of an Instagram reel, extract structured metadata.

If the summary mentions a specific place or restaurant:
1) Use geocode_place to get coordinates (Nominatim / OpenStreetMap)
2) Use tavily_search if the place name is unclear

You may also receive additional reel metadata (TITLE/DESCRIPTION/HASHTAGS/LOCATION_TAG)
that often contains the correct place even when the vision summary is incomplete.

Return ONLY valid JSON in this exact schema:
{
  "place_name": string|null,
  "city": string|null,
  "country": string|null,
  "lat": number|null,
  "lng": number|null,
  "rating": number|null,
  "category": "restaurant"|"beach"|"hotel"|"activity"|"street_food"|"cafe"|"bar",
  "cuisine": string|null,
  "price_range": "budget"|"mid"|"luxury",
  "summary": string,
  "tags": string[]
}

Rules:
- tags should contain 5-8 concise, lowercase tags.
- summary should be a cleaned up version of the input.
- if unknown, use null for optional fields."""


@lru_cache(maxsize=1)
def _build_enrichment_agent():
    from langchain_openai import ChatOpenAI
    from langchain_community.tools.tavily_search import TavilySearchResults

    from backend.integrations.nominatim import nominatim_search

    @tool
    def geocode_place(query: str) -> dict:
        """Search for a place and return key details (coordinates, city, country)."""
        q = (query or "").strip()
        if not q:
            return {"error": "empty query"}
        try:
            results = nominatim_search(q, limit=1, timeout=12.0)
        except Exception as ex:
            return {"error": f"geocode failed: {ex}"}
        if not results:
            return {"error": "Place not found"}
        top = results[0]
        address = top.get("address") or {}
        city = (
            address.get("city")
            or address.get("town")
            or address.get("village")
            or address.get("state_district")
        )
        disp = top.get("display_name") or q
        name = disp.split(",")[0].strip() if disp else q
        return {
            "name": name,
            "formatted_address": disp,
            "rating": None,
            "geometry": {
                "location": {
                    "lat": top.get("lat"),
                    "lng": top.get("lon"),
                }
            },
            "city": city,
            "country": address.get("country"),
        }

    tavily = TavilySearchResults(max_results=2, name="tavily_search")
    tools = [tavily, geocode_place]

    api_key = os.getenv("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("OPENAI_API_KEY is not set")

    llm = ChatOpenAI(model="gpt-4o", temperature=0, api_key=api_key)
    return create_react_agent(llm, tools, prompt=ENRICHMENT_SYSTEM)


def _extract_last_ai_text(messages: list) -> str:
    for msg in reversed(messages or []):
        if isinstance(msg, AIMessage):
            c = msg.content
            if isinstance(c, str) and c.strip():
                return c.strip()
            if isinstance(c, list):
                parts = []
                for block in c:
                    if isinstance(block, dict) and block.get("type") == "text":
                        parts.append(block.get("text", ""))
                if parts:
                    return "\n".join(parts).strip()
    return "{}"


async def run_enrichment_graph(
    vision_summary: str,
    reel_url: Optional[str] = None,
    reel_metadata: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    """
    Run the enrichment ReAct agent and return the same dict shape as legacy enrich_reel_summary.
    Caller applies _parse_enrichment_output / _normalize_enrichment in helpers.
    """
    metadata_text = ""
    if isinstance(reel_metadata, dict):
        metadata_text = reel_metadata.get("metadata_text") or ""

    reel_url_line = f"Reel URL: {reel_url}" if reel_url else ""
    user_text = (
        f"Vision summary: {vision_summary}\n"
        f"{reel_url_line}\n\n"
        f"Additional reel metadata (description/hashtags/location tags):\n{metadata_text}"
    )

    agent = _build_enrichment_agent()
    result = await agent.ainvoke({"messages": [HumanMessage(content=user_text)]})
    messages = result.get("messages") or []
    raw_output = _extract_last_ai_text(messages)
    return {"raw_output": raw_output, "messages": messages}
