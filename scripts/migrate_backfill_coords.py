import argparse
import json
import os
import re
import sys
import time
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional, Tuple

from dotenv import load_dotenv
import requests


def _ensure_project_root_on_path() -> None:
    root = Path(__file__).resolve().parent.parent
    if str(root) not in sys.path:
        sys.path.insert(0, str(root))


def _as_float(value: Any) -> Optional[float]:
    if value is None or value == "":
        return None
    try:
        return float(value)
    except Exception:
        return None


def _is_missing_coordinate(value: Any) -> bool:
    return _as_float(value) is None


def _locationiq_geocode(query: str) -> Dict[str, Any]:
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
    candidates: List[str] = []
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

    for query in candidates:
        geo = _locationiq_geocode(query)
        lat = _as_float(geo.get("lat"))
        lng = _as_float(geo.get("lng"))
        if lat is not None and lng is not None:
            return lat, lng, geo.get("city"), geo.get("country")
    return None, None, None, None


def _pick_place_fields(metadata: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    place_name = metadata.get("enrichment_place_name") or metadata.get("place_name")
    city = metadata.get("enrichment_city") or metadata.get("city")
    country = metadata.get("enrichment_country") or metadata.get("country")
    return place_name, city, country


def _clean_text(value: Any) -> Optional[str]:
    if not isinstance(value, str):
        return None
    text = value.strip()
    if not text:
        return None
    if text.lower() in {"unknown", "none", "n/a", "null"}:
        return None
    return text


def _extract_place_context_from_metadata(metadata: Dict[str, Any]) -> Tuple[Optional[str], Optional[str], Optional[str]]:
    place_name, city, country = _pick_place_fields(metadata)
    if place_name or city:
        return place_name, city, country

    enrichment_json_raw = metadata.get("enrichment_json")
    if isinstance(enrichment_json_raw, str) and enrichment_json_raw.strip():
        try:
            payload = json.loads(enrichment_json_raw)
            if isinstance(payload, dict):
                place_name = _clean_text(payload.get("place_name")) or place_name
                city = _clean_text(payload.get("city")) or city
                country = _clean_text(payload.get("country")) or country
        except Exception:
            pass

    doc_text = _clean_text(metadata.get("doc_text")) or ""
    if doc_text:
        # Old records may contain LOCATION_TAG line from metadata extraction.
        location_match = re.search(r"LOCATION_TAG:\s*(.+)", doc_text, flags=re.IGNORECASE)
        if location_match:
            maybe_place = _clean_text(location_match.group(1))
            if maybe_place and not place_name:
                place_name = maybe_place

    summary_text = _clean_text(metadata.get("summary")) or _clean_text(metadata.get("enrichment_summary"))
    if summary_text and not place_name:
        # Lightweight fallback: capture "in <place>" from summary text.
        in_match = re.search(r"\bin\s+([A-Z][A-Za-z0-9\s'&.-]{2,60})", summary_text)
        if in_match:
            place_name = _clean_text(in_match.group(1))

    return place_name, city, country


def _tavily_search_place_hints(metadata: Dict[str, Any]) -> List[str]:
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        return []

    url = _clean_text(metadata.get("url")) or ""
    summary = _clean_text(metadata.get("summary")) or _clean_text(metadata.get("enrichment_summary")) or ""
    tags = metadata.get("tags") or metadata.get("enrichment_tags") or []
    if not isinstance(tags, list):
        tags = []

    query_parts = ["Find the most likely place name and city from this travel reel context."]
    if url:
        query_parts.append(f"Reel URL: {url}")
    if summary:
        query_parts.append(f"Summary: {summary[:300]}")
    if tags:
        query_parts.append(f"Tags: {', '.join([str(t) for t in tags[:10]])}")
    query = "\n".join(query_parts)

    try:
        resp = requests.post(
            "https://api.tavily.com/search",
            json={
                "api_key": api_key,
                "query": query,
                "search_depth": "basic",
                "include_answer": True,
                "max_results": 3,
            },
            timeout=15,
        )
        resp.raise_for_status()
        payload = resp.json()
    except Exception:
        return []

    candidates: List[str] = []
    answer = _clean_text(payload.get("answer"))
    if answer:
        candidates.append(answer)

    for result in payload.get("results", []) or []:
        if not isinstance(result, dict):
            continue
        for key in ("title", "content"):
            text = _clean_text(result.get(key))
            if text:
                candidates.append(text)

    # Keep small unique set for geocoding attempts.
    unique: List[str] = []
    seen = set()
    for text in candidates:
        candidate = text.strip()
        if candidate.lower() in seen:
            continue
        seen.add(candidate.lower())
        unique.append(candidate[:180])
        if len(unique) >= 5:
            break
    return unique


def _geocode_candidates(candidates: List[str]) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str], Optional[str]]:
    for candidate in candidates:
        lat, lng, city, country = _approximate_coordinates(candidate, None, None)
        if lat is not None and lng is not None:
            return lat, lng, city, country, candidate
    return None, None, None, None, None


def _is_instagram_url(url: str) -> bool:
    u = (url or "").lower()
    return "instagram.com" in u or "instagr.am" in u


def _place_hints_from_ig_extract(ig: Dict[str, Any]) -> List[str]:
    """Build geocode query strings from yt-dlp Instagram metadata (same fields as backend helpers)."""
    if not ig:
        return []
    candidates: List[str] = []
    loc = ig.get("location_tag")
    if isinstance(loc, str) and loc.strip():
        candidates.append(loc.strip())
    title = ig.get("title")
    if isinstance(title, str) and title.strip():
        candidates.append(title.strip()[:200])
    for h in ig.get("hashtags") or []:
        if isinstance(h, str) and len(h.strip()) > 2:
            candidates.append(h.strip().replace("_", " ")[:80])
    desc = ig.get("description") or ""
    if isinstance(desc, str):
        for m in re.finditer(r"#([A-Za-z0-9_]{2,40})", desc[:800]):
            candidates.append(m.group(1).replace("_", " "))
    meta_text = ig.get("metadata_text") or ""
    if isinstance(meta_text, str):
        m = re.search(r"LOCATION_TAG:\s*(.+)", meta_text, flags=re.IGNORECASE | re.DOTALL)
        if m:
            tag = _clean_text(m.group(1).split("\n")[0])
            if tag:
                candidates.append(tag)
    # De-dupe, preserve order, cap length for geocoder.
    seen: set = set()
    out: List[str] = []
    for c in candidates:
        key = c.lower().strip()
        if not key or key in seen:
            continue
        seen.add(key)
        out.append(c[:180])
        if len(out) >= 12:
            break
    return out


def _try_geocode_from_instagram_url(
    reel_url: str,
) -> Tuple[Optional[float], Optional[float], Optional[str], Optional[str], Optional[str]]:
    """
    Fetch public reel metadata via yt-dlp (same as backend) and geocode location hints.
    Returns (lat, lng, city, country, place_label) or Nones if nothing worked.
    """
    _ensure_project_root_on_path()
    from backend.helpers import extract_reel_metadata_with_yt_dlp, strip_igsh_parameter

    url = _clean_text(reel_url)
    if not url or not _is_instagram_url(url):
        return None, None, None, None, None

    cleaned = strip_igsh_parameter(url)
    ig = extract_reel_metadata_with_yt_dlp(cleaned)
    if not ig:
        return None, None, None, None, None

    hints = _place_hints_from_ig_extract(ig)
    return _geocode_candidates(hints)


def _iter_ids(index: Any, namespace: Optional[str], page_size: int) -> Iterable[str]:
    # pinecone-python supports listing IDs in batches.
    for id_batch in index.list(namespace=namespace, limit=page_size):
        for item_id in id_batch:
            yield item_id


def _chunked(items: List[str], size: int) -> Iterable[List[str]]:
    for i in range(0, len(items), size):
        yield items[i : i + size]


def run_migration(
    index_name: str,
    namespace: Optional[str],
    list_page_size: int,
    fetch_batch_size: int,
    dry_run: bool,
    max_records: Optional[int],
    sleep_seconds: float,
    ig_fetch: bool,
    ig_sleep_seconds: float,
) -> None:
    load_dotenv()
    from pinecone import Pinecone

    pinecone_api_key = os.getenv("PINECONE_API_KEY")
    if not pinecone_api_key:
        raise RuntimeError("PINECONE_API_KEY is required (set it in .env or env vars).")

    pc = Pinecone(api_key=pinecone_api_key)
    index = pc.Index(index_name)

    ids: List[str] = []
    for reel_id in _iter_ids(index=index, namespace=namespace, page_size=list_page_size):
        ids.append(reel_id)
        if max_records is not None and len(ids) >= max_records:
            break

    if not ids:
        print("No records found. Nothing to migrate.")
        return

    scanned = 0
    missing = 0
    geocoded = 0
    updated = 0
    skipped_no_place = 0
    skipped_geocode_fail = 0
    recovered_from_metadata = 0
    recovered_from_tavily = 0
    recovered_from_ig = 0

    print(f"Scanning {len(ids)} record(s) from index '{index_name}'...")
    for id_batch in _chunked(ids, fetch_batch_size):
        fetched = index.fetch(ids=id_batch, namespace=namespace)
        vectors = fetched.vectors or {}

        for reel_id, vector in vectors.items():
            scanned += 1
            metadata = vector.metadata or {}

            has_enrichment_missing = _is_missing_coordinate(metadata.get("enrichment_lat")) or _is_missing_coordinate(
                metadata.get("enrichment_lng")
            )
            has_legacy_fields = ("lat" in metadata) or ("lon" in metadata)
            has_legacy_missing = has_legacy_fields and (
                _is_missing_coordinate(metadata.get("lat")) or _is_missing_coordinate(metadata.get("lon"))
            )
            needs_update = has_enrichment_missing or has_legacy_missing
            if not needs_update:
                continue

            missing += 1
            place_name, city, country = _extract_place_context_from_metadata(metadata)

            if (place_name or city) and not (_pick_place_fields(metadata)[0] or _pick_place_fields(metadata)[1]):
                recovered_from_metadata += 1

            if not place_name and not city:
                lat = lng = geocoded_city = geocoded_country = None
                tavily_hit = None

                if ig_fetch:
                    url_for_ig = _clean_text(metadata.get("url")) or ""
                    if _is_instagram_url(url_for_ig):
                        lat, lng, geocoded_city, geocoded_country, ig_hit = _try_geocode_from_instagram_url(
                            url_for_ig
                        )
                        if lat is not None and lng is not None:
                            recovered_from_ig += 1
                            tavily_hit = ig_hit
                        if ig_sleep_seconds > 0:
                            time.sleep(ig_sleep_seconds)

                if lat is None or lng is None:
                    tavily_candidates = _tavily_search_place_hints(metadata)
                    lat, lng, geocoded_city, geocoded_country, tavily_hit = _geocode_candidates(tavily_candidates)
                    if lat is None or lng is None:
                        skipped_no_place += 1
                        continue
                    recovered_from_tavily += 1
            else:
                lat, lng, geocoded_city, geocoded_country = _approximate_coordinates(
                    place_name=place_name,
                    city=city,
                    country=country,
                )
                tavily_hit = None

            if lat is None or lng is None:
                skipped_geocode_fail += 1
                continue

            geocoded += 1
            metadata_updates: Dict[str, Any] = {
                "enrichment_lat": lat,
                "enrichment_lng": lng,
            }

            # Also keep legacy fields in sync if the old schema used them.
            if "lat" in metadata or "lon" in metadata:
                metadata_updates["lat"] = lat
                metadata_updates["lon"] = lng

            if (not metadata.get("enrichment_city")) and geocoded_city:
                metadata_updates["enrichment_city"] = geocoded_city
            if (not metadata.get("enrichment_country")) and geocoded_country:
                metadata_updates["enrichment_country"] = geocoded_country
            if tavily_hit and not metadata.get("enrichment_place_name"):
                metadata_updates["enrichment_place_name"] = tavily_hit

            if dry_run:
                print(f"[DRY RUN] Would update {reel_id} -> lat={lat}, lng={lng}")
            else:
                index.update(id=reel_id, set_metadata=metadata_updates, namespace=namespace)
                updated += 1
                print(f"Updated {reel_id} -> lat={lat}, lng={lng}")
                if sleep_seconds > 0:
                    time.sleep(sleep_seconds)

    print("\nMigration complete.")
    print(f"- scanned: {scanned}")
    print(f"- records with missing coordinates: {missing}")
    print(f"- geocoded successfully: {geocoded}")
    print(f"- updated: {updated}{' (dry run)' if dry_run else ''}")
    print(f"- skipped (missing place context): {skipped_no_place}")
    print(f"- skipped (geocode failed): {skipped_geocode_fail}")
    print(f"- recovered place context from metadata: {recovered_from_metadata}")
    print(f"- recovered place context from Instagram (yt-dlp): {recovered_from_ig}")
    print(f"- recovered place context from Tavily: {recovered_from_tavily}")


def main() -> None:
    parser = argparse.ArgumentParser(
        description="Backfill missing lat/lng metadata in Pinecone records for map visibility."
    )
    parser.add_argument("--index", default="whatsapp-reels", help="Pinecone index name.")
    parser.add_argument("--namespace", default=None, help="Pinecone namespace (default: None).")
    parser.add_argument("--list-page-size", type=int, default=100, help="IDs fetched per list page.")
    parser.add_argument("--fetch-batch-size", type=int, default=100, help="IDs fetched per fetch call.")
    parser.add_argument("--sleep-seconds", type=float, default=0.0, help="Delay between update calls.")
    parser.add_argument("--max-records", type=int, default=None, help="Optional cap for safe incremental runs.")
    parser.add_argument(
        "--dry-run",
        action="store_true",
        help="Show planned updates without writing changes to Pinecone.",
    )
    parser.add_argument(
        "--no-ig-fetch",
        action="store_true",
        help="Skip fetching reel metadata from Instagram via yt-dlp (faster, no network to IG).",
    )
    parser.add_argument(
        "--ig-sleep-seconds",
        type=float,
        default=1.0,
        help="Delay after each Instagram metadata fetch to reduce rate limits (default: 1.0).",
    )
    args = parser.parse_args()

    run_migration(
        index_name=args.index,
        namespace=args.namespace,
        list_page_size=args.list_page_size,
        fetch_batch_size=args.fetch_batch_size,
        dry_run=args.dry_run,
        max_records=args.max_records,
        sleep_seconds=args.sleep_seconds,
        ig_fetch=not args.no_ig_fetch,
        ig_sleep_seconds=args.ig_sleep_seconds,
    )


if __name__ == "__main__":
    main()
