"""LangGraph trip planner: Nominatim, Open-Meteo, Overpass, Amadeus, optional ORS + Pinecone."""

from __future__ import annotations

import asyncio
import json
import time
from datetime import date, timedelta
from typing import Any, Dict, List, Optional, TypedDict

from langchain_core.runnables import RunnableConfig
from langgraph.graph import END, StateGraph

from backend.handlers.common import build_map_points_and_count, build_sources_from_matches, query_index
from backend.integrations import amadeus, open_meteo, openrouteservice, overpass
from backend.integrations.nominatim import nominatim_to_lat_lon_bbox
from backend.intent import _extract_destination
from backend.models import AgenticQueryResponse, IntentDetectionResult, MapPoint, QueryCard, QueryIntent, QueryMeta, ReelResult
from backend.observability import async_time_block, get_log, record_timing
from backend.trip_utils import build_day_buckets, extract_trip_days


class TripPlannerState(TypedDict, total=False):
    query: str
    intent_entities: Dict[str, Any]
    top_k: int
    days: int
    destination_lat: Optional[float]
    destination_lon: Optional[float]
    destination_label: Optional[str]
    origin_label: Optional[str]
    origin_lat: Optional[float]
    origin_lon: Optional[float]
    weather_compact: Any
    poi_candidates: List[Dict[str, Any]]
    flight_summaries: List[Dict[str, Any]]
    ground_km: Optional[float]
    transit_stops: List[Dict[str, Any]]
    debug_notes: List[str]
    selected: List[ReelResult]
    map_points: List[MapPoint]
    day_buckets: Dict[int, List[ReelResult]]
    geocoded_count: int
    itinerary_cards: List[QueryCard]
    agentic_response: Optional[Dict[str, Any]]


def _destination_from_state(state: TripPlannerState) -> str:
    ent = state.get("intent_entities") or {}
    dest = ent.get("destination")
    if dest and str(dest).strip():
        return str(dest).strip()
    q = state.get("query") or ""
    extracted = _extract_destination(q)
    if extracted:
        return extracted
    return "Paris"


def _origin_from_state(state: TripPlannerState) -> str:
    ent = state.get("intent_entities") or {}
    o = ent.get("origin")
    if o and str(o).strip():
        return str(o).strip()
    return "New York"


def _departure_date_str(state: TripPlannerState) -> Optional[str]:
    ent = state.get("intent_entities") or {}
    for key in ("departure_date", "dates", "travel_dates"):
        v = ent.get(key)
        if v and str(v).strip():
            s = str(v).strip()
            if len(s) >= 10 and s[4] == "-" and s[7] == "-":
                return s[:10]
    return None


async def node_resolve_destination(state: TripPlannerState) -> Dict[str, Any]:
    async with async_time_block("trip_node_resolve", "trip_planner_graph", node="resolve"):
        days = extract_trip_days(state.get("query") or "")
        dest_q = _destination_from_state(state)
        notes: List[str] = []
        lat: Optional[float] = None
        lon: Optional[float] = None
        label: Optional[str] = dest_q
        try:
            lat, lon, label, _bbox = await asyncio.to_thread(nominatim_to_lat_lon_bbox, dest_q)
            if lat is None:
                notes.append("nominatim_miss_destination")
        except Exception as ex:
            notes.append(f"nominatim_error:{ex}")

        olab = _origin_from_state(state)
        o_lat = o_lon = None
        try:
            o_lat, o_lon, _, _ = await asyncio.to_thread(nominatim_to_lat_lon_bbox, olab)
        except Exception:
            pass

        out: Dict[str, Any] = {
            "days": days,
            "destination_label": label or dest_q,
            "origin_label": olab,
            "debug_notes": notes,
        }
        if lat is not None:
            out["destination_lat"] = lat
        if lon is not None:
            out["destination_lon"] = lon
        if o_lat is not None:
            out["origin_lat"] = o_lat
        if o_lon is not None:
            out["origin_lon"] = o_lon
        get_log().info(
            "trip_graph_node_outcome",
            node="resolve",
            destination_label=out.get("destination_label"),
            has_destination_coords=lat is not None and lon is not None,
        )
        return out


async def node_gather_context(state: TripPlannerState, config: RunnableConfig) -> Dict[str, Any]:
    async with async_time_block("trip_node_gather", "trip_planner_graph", node="gather"):
        conf = config.get("configurable") or {}
        index = conf.get("index")
        query_embedding = conf.get("query_embedding")
        top_k = int(state.get("top_k") or 5)
        days = int(state.get("days") or 2)
        query = state.get("query") or ""

        debug_notes = list(state.get("debug_notes") or [])
        lat = state.get("destination_lat")
        lon = state.get("destination_lon")

        async def _run_timed(name: str, coro: Any) -> Any:
            t0 = time.perf_counter()
            try:
                return await coro
            finally:
                ms = (time.perf_counter() - t0) * 1000
                record_timing(f"trip_gather_{name}", ms)
                get_log().info(
                    "trip_gather_subtask",
                    subtask=name,
                    duration_ms=round(ms, 2),
                )

        # Pinecone
        selected: List[ReelResult] = []
        t_pc = time.perf_counter()
        if index is not None and query_embedding is not None:
            try:
                res = query_index(index=index, query_embedding=query_embedding, top_k=top_k)
                matches = res.matches or []
                sources = build_sources_from_matches(matches)
                selected = sources[:top_k]
            except Exception as ex:
                debug_notes.append(f"pinecone:{ex}")
        record_timing("trip_gather_pinecone", (time.perf_counter() - t_pc) * 1000)

        day_buckets = build_day_buckets(selected, days)
        map_points, geocoded_count = build_map_points_and_count(selected)
        reel_to_day_sequence: Dict[str, tuple] = {}
        for day in range(1, days + 1):
            seq = 1
            for source in day_buckets[day]:
                reel_to_day_sequence[source.reel_id] = (f"Day {day}", seq)
                seq += 1
        for point in map_points:
            group, sequence = reel_to_day_sequence.get(point.reel_id, ("Day 1", 1))
            point.group = group
            point.sequence = sequence

        itinerary_cards: List[QueryCard] = []
        for day in range(1, days + 1):
            items = day_buckets[day]
            if not items:
                itinerary_cards.append(
                    QueryCard(
                        card_type="itinerary_day",
                        title=f"Day {day}",
                        summary="No strong matches yet for this day. Try adding more reels for better planning.",
                        metadata={"places": []},
                    )
                )
                continue
            places = []
            for source in items:
                enrichment = source.enrichment
                places.append(
                    {
                        "place_name": enrichment.place_name if enrichment else "Unknown place",
                        "city": enrichment.city if enrichment else None,
                        "country": enrichment.country if enrichment else None,
                        "source_url": source.url,
                    }
                )
            itinerary_cards.append(
                QueryCard(
                    card_type="itinerary_day",
                    title=f"Day {day}",
                    summary=f"{len(items)} stop(s) planned from your saved reels.",
                    metadata={"places": places},
                )
            )

        weather_compact: Any = None
        poi_candidates: List[Dict[str, Any]] = []
        flight_summaries: List[Dict[str, Any]] = []
        transit_stops: List[Dict[str, Any]] = []
        ground_km: Optional[float] = None

        async def _weather():
            async def _inner():
                if lat is None or lon is None:
                    return None
                raw = await asyncio.to_thread(
                    open_meteo.fetch_forecast_daily, float(lat), float(lon), days=max(days, 3)
                )
                return open_meteo.summarize_forecast_for_prompt(raw, max_days=max(days, 5))

            return await _run_timed("weather", _inner())

        async def _pois():
            async def _inner():
                if lat is None or lon is None:
                    return []
                return await asyncio.to_thread(overpass.tourism_food_parks_around, float(lat), float(lon))

            return await _run_timed("pois", _inner())

        async def _transit():
            async def _inner():
                if lat is None or lon is None:
                    return []
                return await asyncio.to_thread(overpass.railway_bus_stops_around, float(lat), float(lon))

            return await _run_timed("transit", _inner())

        async def _flights():
            async def _inner():
                dep_date = _departure_date_str(state) or (date.today() + timedelta(days=21)).isoformat()
                origin = _origin_from_state(state)
                dest = _destination_from_state(state)
                try:
                    orows = await asyncio.to_thread(amadeus.search_airport_codes, origin, max_results=2)
                    drows = await asyncio.to_thread(amadeus.search_airport_codes, dest, max_results=2)
                    oiata = amadeus.pick_iata_from_location_results(orows)
                    diata = amadeus.pick_iata_from_location_results(drows)
                    if not oiata or not diata:
                        debug_notes.append("flight_skip_no_iata")
                        return []
                    offers = await asyncio.to_thread(amadeus.flight_offers_search, oiata, diata, dep_date)
                    return amadeus.compact_offer_summary(offers, limit=3)
                except Exception as ex:
                    debug_notes.append(f"flight_error:{ex}")
                    return []

            return await _run_timed("flights", _inner())

        async def _ors():
            async def _inner():
                olat = state.get("origin_lat")
                olon = state.get("origin_lon")
                if None in (olat, olon, lat, lon):
                    return None
                gj = await asyncio.to_thread(
                    openrouteservice.directions_geojson,
                    float(olon),
                    float(olat),
                    float(lon),
                    float(lat),
                )
                return openrouteservice.summarize_route_distance_km(gj)

            return await _run_timed("ors", _inner())

        w_task = asyncio.create_task(_weather())
        p_task = asyncio.create_task(_pois())
        t_task = asyncio.create_task(_transit())
        f_task = asyncio.create_task(_flights())
        r_task = asyncio.create_task(_ors())

        weather_compact, poi_candidates, transit_stops, flight_summaries, ground_km = await asyncio.gather(
            w_task, p_task, t_task, f_task, r_task
        )

        get_log().info(
            "trip_graph_node_outcome",
            node="gather",
            selected_count=len(selected),
            geocoded_count=geocoded_count,
        )

        return {
            "selected": selected,
            "map_points": map_points,
            "day_buckets": day_buckets,
            "geocoded_count": geocoded_count,
            "itinerary_cards": itinerary_cards,
            "weather_compact": weather_compact,
            "poi_candidates": poi_candidates or [],
            "flight_summaries": flight_summaries or [],
            "transit_stops": transit_stops or [],
            "ground_km": ground_km,
            "debug_notes": debug_notes,
        }


async def node_compose(state: TripPlannerState, config: RunnableConfig) -> Dict[str, Any]:
    async with async_time_block("trip_node_compose", "trip_planner_graph", node="compose"):
        conf = config.get("configurable") or {}
        client = conf.get("openai_client")
        intent_result: IntentDetectionResult = conf.get("intent_result")
        
        days = int(state.get("days") or 2)
        query = state.get("query") or ""
        selected = state.get("selected") or []
        map_points = state.get("map_points") or []
        itinerary_cards = list(state.get("itinerary_cards") or [])
        
        weather_compact = state.get("weather_compact")
        poi_candidates = state.get("poi_candidates") or []
        flight_summaries = state.get("flight_summaries") or []
        transit_stops = state.get("transit_stops") or []
        ground_km = state.get("ground_km")
        dest_label = state.get("destination_label") or ""
        day_buckets = state.get("day_buckets") or {}
        
        reel_place_refs: List[Dict[str, Any]] = []
        reel_seq = 1
        for day in range(1, days + 1):
            for source in day_buckets.get(day) or []:
                enrichment = source.enrichment
                reel_place_refs.append(
                    {
                        "ref": f"P{reel_seq}",
                        "day": day,
                        "place": enrichment.place_name if enrichment else None,
                        "city": enrichment.city if enrichment else None,
                        "country": enrichment.country if enrichment else None,
                        "source_url": source.url,
                    }
                )
                reel_seq += 1
        
        api_payload = {
            "destination": dest_label,
            "weather_daily": weather_compact,
            "sample_pois": poi_candidates[:12],
            "flight_options": flight_summaries,
            "transit_stops_near_destination": transit_stops[:12],
            "road_distance_km_ors": ground_km,
            "transport_data_sources": {
                "transit_stops": "OpenStreetMap Overpass railway/bus stops near destination",
                "road_distance": "OpenRouteService driving distance between origin and destination",
                "flight_options": "Amadeus flight offers (when available)",
            },
            "saved_reel_place_refs": reel_place_refs[:20],
        }
        
        narrative = ""
        compose_llm_ok = False
        if client is not None:
            try:
                resp = client.chat.completions.create(
                    model="gpt-4o-mini",
                    temperature=0.25,
                    messages=[
                        {
                            "role": "system",
                            "content": (
                                "You are a travel planner. Write a helpful trip outline using ONLY the JSON context provided. "
                                "Output plain Markdown prose only (headings, bullets, bold). "
                                "Never output raw JSON, code fences, or key-value dumps as the answer. "
                                "Do not invent specific prices, flight numbers, or schedules not in the JSON. "
                                "In Getting there, explicitly use transport integrations present in JSON: Overpass transit stops, ORS road distance, and Amadeus flights when available. "
                                "Do not give generic 'use apps' advice by itself; include concrete stations/stops and distance context from JSON first, then mention app booking only for live schedules. "
                                "In Your saved spots, refer to each spot with its ref ID (P1/P2/...) and city so it is unambiguous which place is which. "
                                "Tie in saved reels when reel places appear in the context. "
                                "Use short sections: Overview, Weather, Getting there, Around the destination, Your saved spots. "
                                "End with one short follow-up question."
                            ),
                        },
                        {
                            "role": "user",
                            "content": (
                                f"User query: {query}\n"
                                f"Trip days: {days}\n"
                                f"Saved reel places (from vector DB, labeled): {json.dumps(reel_place_refs[:20], ensure_ascii=True)}\n"
                                f"API context JSON: {json.dumps(api_payload, default=str, ensure_ascii=True)[:12000]}"
                            ),
                        },
                    ],
                )
                narrative = (resp.choices[0].message.content or "").strip()
                compose_llm_ok = bool(narrative)
            except Exception as ex:
                get_log().exception(
                    "trip_compose_llm_failed",
                    node="compose",
                    agent="trip_planner_graph",
                    error_type=type(ex).__name__,
                    error_message=str(ex)[:500],
                )
                narrative = ""
        
        if not narrative:
            narrative = (
                f"Draft {days}-day plan for {dest_label}. "
                + ("Weather data included where available. " if weather_compact else "")
                + ("Stops from your saved reels are mapped by day when available. " if selected else "Save reels to personalize this itinerary. ")
            )
        
        extra_cards: List[QueryCard] = []
        if weather_compact:
            extra_cards.append(
                QueryCard(
                    card_type="weather_summary",
                    title="Weather snapshot",
                    summary="Forecast from Open-Meteo (daily).",
                    metadata={"forecast": weather_compact[:5]},
                )
            )
        if flight_summaries:
            extra_cards.append(
                QueryCard(
                    card_type="transport_flights",
                    title="Flight options (Amadeus)",
                    summary="Sample offers from the developer API — verify live prices before booking.",
                    metadata={"offers": flight_summaries},
                )
            )
        if transit_stops:
            extra_cards.append(
                QueryCard(
                    card_type="transit_stops",
                    title="Nearby stations & stops (OpenStreetMap)",
                    summary="Use local transit apps for live schedules.",
                    metadata={"stops": transit_stops[:10]},
                )
            )
        if poi_candidates:
            extra_cards.append(
                QueryCard(
                    card_type="poi_suggestions",
                    title="Things nearby (OSM)",
                    summary="Points of interest near the destination center.",
                    metadata={"pois": poi_candidates[:10]},
                )
            )
        
        cards = itinerary_cards + extra_cards
        
        meta = QueryMeta(
            confidence=intent_result.confidence if intent_result else 0.0,
            debug_route="trip_planner_graph",
            clarification_needed=intent_result.clarification_needed if intent_result else False,
            geocoded_on_the_fly=state.get("geocoded_count") or 0,
            result_count=len(selected),
            map_points_count=len(map_points),
            applied_filters={
                "trip_days": days,
                "apis_used": ["nominatim", "open_meteo", "overpass", "amadeus_optional", "ors_optional"],
                "debug_notes": state.get("debug_notes") or [],
            },
            skip_conversational_rewrite=True,
        )
        
        agentic = AgenticQueryResponse(
            intent=QueryIntent.trip_planning,
            map_points=map_points,
            cards=cards,
            narrative=narrative,
            sources=selected,
            meta=meta,
        )
        
        get_log().info(
            "trip_graph_node_outcome",
            node="compose",
            narrative_chars=len(narrative or ""),
            compose_llm_ok=compose_llm_ok,
            cards_count=len(cards),
        )
        return {"agentic_response": agentic.model_dump(mode="json")}


def build_trip_planner_graph():
    g = StateGraph(TripPlannerState)
    g.add_node("resolve", node_resolve_destination)
    g.add_node("gather", node_gather_context)
    g.add_node("compose", node_compose)
    g.set_entry_point("resolve")
    g.add_edge("resolve", "gather")
    g.add_edge("gather", "compose")
    g.add_edge("compose", END)
    return g.compile()


_GRAPH = None


def _compiled():
    global _GRAPH
    if _GRAPH is None:
        _GRAPH = build_trip_planner_graph()
    return _GRAPH


async def run_trip_planner_graph(
    *,
    query: str,
    index: Any,
    query_embedding: List[float],
    top_k: int,
    intent_result: IntentDetectionResult,
    client: Any,
) -> AgenticQueryResponse:
    graph = _compiled()
    init: TripPlannerState = {
        "query": query,
        "intent_entities": intent_result.entities.model_dump(exclude_none=True),
        "top_k": top_k,
    }
    cfg: RunnableConfig = {
        "configurable": {
            "index": index,
            "query_embedding": query_embedding,
            "openai_client": client,
            "intent_result": intent_result,
        }
    }
    out = await graph.ainvoke(init, cfg)
    raw = out.get("agentic_response")
    if not raw:
        raise RuntimeError("trip planner graph produced no response")
    return AgenticQueryResponse.model_validate(raw)
