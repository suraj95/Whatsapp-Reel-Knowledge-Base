import base64
import glob
import json
import os
import re
import subprocess
import tempfile
from functools import lru_cache
from typing import Any, Dict, List, Optional, Tuple

from openai import OpenAI
import requests
from yt_dlp import YoutubeDL


def _locationiq_geocode(query: str) -> Dict:
    maps_key = os.getenv("LOCATIONIQ_API_KEY")
    if not maps_key or not query:
        return {}
    try:
        resp = requests.get(
            "https://us1.locationiq.com/v1/search.php",
            params={
                "key": maps_key,
                "q": query,
                "format": "json",
                "limit": 1,
                "addressdetails": 1,
            },
            timeout=8,
        )
        resp.raise_for_status()
        results = resp.json()
        if isinstance(results, list) and results:
            top = results[0]
            address = top.get("address", {})
            city = (
                address.get("city")
                or address.get("town")
                or address.get("village")
                or address.get("state_district")
            )
            return {
                "lat": top.get("lat"),
                "lng": top.get("lon"),
                "city": city,
                "country": address.get("country"),
            }
    except Exception:
        return {}
    return {}


def _approximate_coordinates(
    place_name: Optional[str], city: Optional[str], country: Optional[str]
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str]]:
    """
    Best-effort geocoding fallback for enrichment outputs missing lat/lng.
    """
    candidates = []
    if place_name and city and country:
        candidates.append(f"{place_name}, {city}, {country}")
    if place_name and city:
        candidates.append(f"{place_name}, {city}")
    if place_name and country:
        candidates.append(f"{place_name}, {country}")
    if place_name:
        candidates.append(place_name)
    if city and country:
        candidates.append(f"{city}, {country}")
    if city:
        candidates.append(city)

    def _as_float(value):
        if value is None or value == "":
            return None
        try:
            return float(value)
        except Exception:
            return None

    for query in candidates:
        geo = _locationiq_geocode(query)
        lat = _as_float(geo.get("lat"))
        lng = _as_float(geo.get("lng"))
        if lat is not None and lng is not None:
            return lat, lng, geo.get("city"), geo.get("country")
    return None, None, None, None


def extract_reel_metadata_with_yt_dlp(reel_url: str) -> Dict[str, Any]:
    """
    Best-effort metadata extraction from the reel URL using yt-dlp.
    This typically includes title + description + tags/hashtags.
    """
    if not reel_url:
        return {}

    ydl_opts = {
        "quiet": True,
        "no_warnings": True,
        # We only need metadata; do not download the file here.
        "skip_download": True,
    }

    try:
        with YoutubeDL(ydl_opts) as ydl:
            info = ydl.extract_info(reel_url, download=False)
    except Exception:
        return {}

    if not isinstance(info, dict):
        return {}

    title = info.get("title")
    description = info.get("description") or info.get("full_description")

    # yt-dlp may return location in various fields depending on extractor.
    location_tag = (
        info.get("location")
        or info.get("location_name")
        or info.get("place")
        or info.get("region")
    )

    tags = info.get("tags") or []
    if not isinstance(tags, list):
        tags = []

    # Extract hashtags from description and tags.
    hashtags: List[str] = []
    if isinstance(description, str):
        hashtags.extend(re.findall(r"#([A-Za-z0-9_]+)", description))
    for t in tags:
        if isinstance(t, str) and t.strip():
            hashtags.append(t.strip().lstrip("#"))

    # Keep the prompt payload small.
    def _truncate(s: str, n: int) -> str:
        s = (s or "").strip()
        if len(s) <= n:
            return s
        return s[: n - 3].rstrip() + "..."

    description_t = _truncate(description, 1800)
    title_t = _truncate(title, 250)
    hashtags_u = list(dict.fromkeys([h.lower() for h in hashtags if h and h.strip()]))
    hashtags_t = _truncate(", ".join(hashtags_u[:30]), 800)

    metadata_text_parts = []
    if title_t:
        metadata_text_parts.append(f"TITLE: {title_t}")
    if description_t:
        metadata_text_parts.append(f"DESCRIPTION: {description_t}")
    if hashtags_t:
        metadata_text_parts.append(f"HASHTAGS: {hashtags_t}")
    if location_tag and isinstance(location_tag, str) and location_tag.strip():
        metadata_text_parts.append(f"LOCATION_TAG: {_truncate(location_tag, 250)}")

    return {
        "title": title_t,
        "description": description_t,
        "hashtags": hashtags_u[:30],
        "location_tag": location_tag,
        "metadata_text": "\n".join(metadata_text_parts).strip(),
    }


def _extract_frames(video_path: str, frames_dir: str) -> List[str]:
    os.makedirs(frames_dir, exist_ok=True)
    # 0.5 fps = 1 frame every 2 seconds
    frame_pattern = os.path.join(frames_dir, "frame_%04d.jpg")
    cmd = [
        "ffmpeg",
        "-i",
        video_path,
        "-vf",
        "fps=0.5",
        "-qscale:v",
        "2",
        frame_pattern,
        "-hide_banner",
        "-loglevel",
        "error",
    ]
    try:
        subprocess.run(cmd, check=True)
    except subprocess.CalledProcessError:
        # ffmpeg could not read the file (corrupt / not actually a video)
        return []

    frame_paths = sorted(glob.glob(os.path.join(frames_dir, "frame_*.jpg")))
    return frame_paths


def summarize_video_with_gpt4o(client: OpenAI, reel_url: str) -> str:
    """
    Download the reel video, extract frames (1 every 2 seconds) with ffmpeg,
    send a subset of those frames to GPT-4o Vision, and return a short summary.

    Requires ffmpeg to be installed and available on PATH.
    """
    with tempfile.TemporaryDirectory() as tmpdir:
        video_path = os.path.join(tmpdir, "reel.mp4")
        frames_dir = os.path.join(tmpdir, "frames")

        # Download using yt-dlp (supports Instagram, etc.)
        ydl_opts = {
            "outtmpl": video_path,
            "format": "mp4/bestvideo+bestaudio/best",
            "quiet": True,
            "no_warnings": True,
        }

        frame_paths: List[str] = []
        try:
            with YoutubeDL(ydl_opts) as ydl:
                ydl.download([reel_url])
            frame_paths = _extract_frames(video_path, frames_dir)
        except Exception as e:
            # Surface a clear error so the API layer can report details back
            raise RuntimeError(f"Video download failed for this URL: {e}") from e

        if not frame_paths:
            # Fall back to URL-only summary if no frames extracted
            resp = client.responses.create(
                model="gpt-4o",
                input=[
                    {
                        "role": "user",
                        "content": [
                            {
                                "type": "input_text",
                                "text": (
                                    "You are helping me build a personal knowledge base from short social media reels.\n"
                                    "Please infer the content of this reel/video and summarize it in 2-3 short sentences, "
                                    "focusing on practical information (e.g. destination, food place, tips, prices, etc.).\n\n"
                                    f"Reel URL: {reel_url}"
                                ),
                            }
                        ],
                    }
                ],
            )
            return resp.output[0].content[0].text.strip()

        # Limit number of frames sent to GPT-4o to keep request light
        max_frames = 8
        selected_frames = frame_paths[:max_frames]

        image_contents = []
        for frame_path in selected_frames:
            with open(frame_path, "rb") as f:
                b64 = base64.b64encode(f.read()).decode("utf-8")
            image_contents.append(
                {
                    "type": "input_image",
                    "image_url": f"data:image/jpeg;base64,{b64}",
                }
            )

        resp = client.responses.create(
            model="gpt-4o",
            input=[
                {
                    "role": "user",
                    "content": [
                        {
                            "type": "input_text",
                            "text": (
                                "You are helping me build a personal knowledge base from short social media reels.\n"
                                "Look at these frames extracted from a single reel (1 frame every ~2 seconds) "
                                "and summarize what the reel is about in 2-3 short sentences. "
                                "Focus on destination, venue/restaurant names, activities, tips, and any prices or recommendations."
                            ),
                        },
                        *image_contents,
                    ],
                }
            ],
        )
        return resp.output[0].content[0].text.strip()


def auto_tag_text(client: OpenAI, text: str) -> List[str]:
    """
    Ask the LLM for a small set of tags like travel / restaurant / hotel / street-food / tips, etc.
    """
    resp = client.chat.completions.create(
        model="gpt-4o-mini",
        messages=[
            {
                "role": "system",
                "content": (
                    "You are a tagging assistant. Given a reel transcript or summary, "
                    "return 3-5 short tags (like 'travel', 'restaurant', 'hotel', "
                    "'street-food', 'Bali', 'Goa', 'budget', etc.) as a comma-separated list."
                ),
            },
            {"role": "user", "content": text},
        ],
        temperature=0.3,
    )
    raw = resp.choices[0].message.content.strip()
    tags = [t.strip() for t in raw.split(",") if t.strip()]
    # de-duplicate and normalize
    unique = list(dict.fromkeys([t.lower() for t in tags]))
    return unique


def embed_text(client: OpenAI, text: str) -> List[float]:
    resp = client.embeddings.create(
        model="text-embedding-3-small",
        input=text,
    )
    return resp.data[0].embedding


@lru_cache(maxsize=1)
def _get_enrichment_executor():
    try:
        from langchain.agents import create_agent
        from langchain_community.tools.tavily_search import TavilySearchResults
        from langchain_core.tools import tool
    except Exception as e:
        raise RuntimeError(
            "Missing enrichment dependencies. Install langchain and "
            "langchain-community."
        ) from e

    @tool
    def search_place_details(query: str) -> dict:
        """Search for a place and return key details."""
        maps_key = os.getenv("LOCATIONIQ_API_KEY")
        if not maps_key:
            return {"error": "LOCATIONIQ_API_KEY is not set"}
        try:
            resp = requests.get(
                "https://us1.locationiq.com/v1/search.php",
                params={
                    "key": maps_key,
                    "q": query,
                    "format": "json",
                    "limit": 1,
                    "addressdetails": 1,
                },
                timeout=10,
            )
            resp.raise_for_status()
            results = resp.json()
            if isinstance(results, list) and results:
                top = results[0]
                address = top.get("address", {})
                city = (
                    address.get("city")
                    or address.get("town")
                    or address.get("village")
                    or address.get("state_district")
                )
                return {
                    "name": top.get("display_name", "").split(",")[0].strip() or query,
                    "formatted_address": top.get("display_name"),
                    "rating": None,  # LocationIQ geocoding does not return ratings.
                    "geometry": {
                        "location": {
                            "lat": top.get("lat"),
                            "lng": top.get("lon"),
                        }
                    },
                    "city": city,
                    "country": address.get("country"),
                }
            return {"error": "Place not found"}
        except Exception as ex:
            return {"error": f"Place lookup failed: {ex}"}

    tools = [
        TavilySearchResults(max_results=2),
        search_place_details,
    ]

    system_prompt = """You are an enrichment agent for a travel/food reel database.
Given a raw vision summary of an Instagram reel, extract structured metadata.

If the summary mentions a specific place or restaurant:
1) Use search_place_details to get coordinates and rating
2) Use TavilySearchResults if the place name is unclear

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

    return create_agent(
        model="openai:gpt-4o",
        tools=tools,
        system_prompt=system_prompt,
    )


def _normalize_enrichment(payload: Dict, vision_summary: str) -> Dict:
    category = payload.get("category") or "activity"
    if category not in {
        "restaurant",
        "beach",
        "hotel",
        "activity",
        "street_food",
        "cafe",
        "bar",
    }:
        category = "activity"

    price_range = payload.get("price_range") or "mid"
    if price_range not in {"budget", "mid", "luxury"}:
        price_range = "mid"

    tags = payload.get("tags") or []
    if not isinstance(tags, list):
        tags = []
    tags = [str(t).strip().lower() for t in tags if str(t).strip()]
    tags = list(dict.fromkeys(tags))[:8]

    def _as_float(value):
        if value is None or value == "":
            return None
        try:
            return float(value)
        except Exception:
            return None

    place_name = payload.get("place_name")
    if isinstance(place_name, str) and place_name.strip().lower() in {"unknown", "n/a", "none", ""}:
        place_name = None
    city = payload.get("city")
    country = payload.get("country")
    lat = _as_float(payload.get("lat"))
    lng = _as_float(payload.get("lng"))

    # If model output has location context but no coordinates, geocode best-effort.
    if lat is None or lng is None:
        approx_lat, approx_lng, approx_city, approx_country = _approximate_coordinates(
            place_name=place_name,
            city=city,
            country=country,
        )
        if lat is None:
            lat = approx_lat
        if lng is None:
            lng = approx_lng
        if not city and approx_city:
            city = approx_city
        if not country and approx_country:
            country = approx_country

    return {
        "place_name": place_name,
        "city": city,
        "country": country,
        "lat": lat,
        "lng": lng,
        "rating": _as_float(payload.get("rating")),
        "category": category,
        "cuisine": payload.get("cuisine"),
        "price_range": price_range,
        "summary": (payload.get("summary") or vision_summary).strip(),
        "tags": tags,
    }


def _parse_enrichment_output(raw_output: str, vision_summary: str) -> Dict:
    cleaned = (raw_output or "").strip()
    if cleaned.startswith("```"):
        cleaned = re.sub(r"^```(?:json)?\s*", "", cleaned, flags=re.IGNORECASE)
        cleaned = re.sub(r"\s*```$", "", cleaned)

    try:
        parsed = json.loads(cleaned)
    except json.JSONDecodeError:
        # Fallback 1: parse first JSON object-like block from text.
        start = cleaned.find("{")
        end = cleaned.rfind("}")
        if start == -1 or end == -1 or end <= start:
            parsed = None
        else:
            try:
                parsed = json.loads(cleaned[start : end + 1])
            except json.JSONDecodeError:
                parsed = None

        # Fallback 2: if the model returned nested {"enrichment": {...}} or {"output": {...}}
        if parsed is None:
            parsed = {}
        elif isinstance(parsed, dict):
            if isinstance(parsed.get("enrichment"), dict):
                parsed = parsed["enrichment"]
            elif isinstance(parsed.get("output"), dict):
                parsed = parsed["output"]
        else:
            parsed = {}
    return _normalize_enrichment(parsed, vision_summary)


async def enrich_reel_summary(
    vision_summary: str,
    reel_url: Optional[str] = None,
    reel_metadata: Optional[Dict[str, Any]] = None,
) -> Dict:
    """
    Enrich a raw visual summary into structured travel/food metadata.
    """
    executor = _get_enrichment_executor()
    metadata_text = ""
    if isinstance(reel_metadata, dict):
        metadata_text = reel_metadata.get("metadata_text") or ""

    # Optional: include reel URL for better grounding.
    reel_url_line = f"Reel URL: {reel_url}" if reel_url else ""

    result = await executor.ainvoke(
        {
            "messages": [
                {
                    "role": "user",
                    "content": (
                        f"Vision summary: {vision_summary}\n"
                        f"{reel_url_line}\n\n"
                        f"Additional reel metadata (description/hashtags/location tags):\n{metadata_text}"
                    ),
                }
            ]
        }
    )
    raw_output = "{}"
    messages = result.get("messages", [])
    if messages:
        last_message = messages[-1]
        content = getattr(last_message, "content", "")
        if isinstance(content, str):
            raw_output = content
        elif isinstance(content, list):
            text_parts = []
            for item in content:
                if isinstance(item, dict):
                    text = item.get("text")
                    if text:
                        text_parts.append(str(text))
            if text_parts:
                raw_output = "\n".join(text_parts)
    return _parse_enrichment_output(raw_output, vision_summary)


def format_enrichment_for_metadata(enrichment: Dict) -> Tuple[Dict, str]:
    """
    Keep metadata Pinecone-friendly and also preserve the full JSON payload.
    """
    metadata = {
        "enrichment_place_name": enrichment.get("place_name"),
        "enrichment_city": enrichment.get("city"),
        "enrichment_country": enrichment.get("country"),
        "enrichment_lat": enrichment.get("lat"),
        "enrichment_lng": enrichment.get("lng"),
        "enrichment_rating": enrichment.get("rating"),
        "enrichment_category": enrichment.get("category"),
        "enrichment_cuisine": enrichment.get("cuisine"),
        "enrichment_price_range": enrichment.get("price_range"),
        "enrichment_summary": enrichment.get("summary"),
        "enrichment_tags": [str(t) for t in enrichment.get("tags", []) if str(t).strip()],
    }
    # Pinecone metadata does not accept null values, so omit None fields.
    metadata = {k: v for k, v in metadata.items() if v is not None}
    payload = json.dumps(enrichment)
    return metadata, payload

