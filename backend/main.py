import os
import uuid
from typing import List

from fastapi import FastAPI, HTTPException
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone

from .helpers import (
    auto_tag_text,
    embed_text,
    extract_reel_metadata_with_yt_dlp,
    enrich_reel_summary,
    format_enrichment_for_metadata,
    summarize_video_with_gpt4o,
)
from .models import (
    AddReelRequest,
    AddReelResponse,
    EnrichRequest,
    EnrichResponse,
    EnrichmentData,
    QueryRequest,
    QueryResponse,
    ReelResult,
)

load_dotenv()

OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")
if not OPENAI_API_KEY:
    raise RuntimeError("Set OPENAI_API_KEY in your environment (e.g. in a .env file).")

PINECONE_API_KEY = os.getenv("PINECONE_API_KEY")
if not PINECONE_API_KEY:
    raise RuntimeError("Set PINECONE_API_KEY in your environment (e.g. in a .env file).")

client = OpenAI(api_key=OPENAI_API_KEY)

# ---- Vector DB (Pinecone) ----
pc = Pinecone(api_key=PINECONE_API_KEY)
INDEX_NAME = "whatsapp-reels"

existing_indexes = [idx.name for idx in pc.list_indexes()]
if INDEX_NAME not in existing_indexes:
    raise RuntimeError(
        f"Pinecone index '{INDEX_NAME}' not found. "
        "Create it in the Pinecone console with dimension=1536 and metric='cosine'."
    )

index = pc.Index(INDEX_NAME)

# ---- FastAPI app ----
app = FastAPI(title="WhatsApp Reel Knowledge Base")


@app.post("/reels", response_model=AddReelResponse)
async def add_reel(payload: AddReelRequest):
    # 1. Summarize the reel with GPT-4o (vision-capable) using downloaded frames
    try:
        summary = summarize_video_with_gpt4o(client, payload.url)
    except Exception as e:
        # Most likely: private / blocked / unsupported URL or yt-dlp/ffmpeg error
        message = str(e)
        if not message:
            message = repr(e)
        # Truncate overly long errors
        if len(message) > 500:
            message = message[:500] + "..."
        raise HTTPException(
            status_code=400,
            detail=f"Could not process this video link. Reason: {message}",
        )

    # 3. Auto tag
    auto_tags = auto_tag_text(client, summary)
    if payload.manual_tags:
        auto_tags = list(
            dict.fromkeys(auto_tags + [t.lower() for t in payload.manual_tags])
        )

    # 3b. Enrich summary using an agentic flow
    # Also extract text metadata from the reel URL (description/hashtags/location tag)
    # to improve location detection when the vision summary alone is incomplete.
    reel_metadata = extract_reel_metadata_with_yt_dlp(payload.url)
    enrichment = await enrich_reel_summary(
        summary,
        reel_url=payload.url,
        reel_metadata=reel_metadata,
    )
    enrichment_metadata, enrichment_json = format_enrichment_for_metadata(enrichment)

    # 4. Build embedding using summary + tags
    doc_text = (
        f"URL: {payload.url}\n"
        f"SUMMARY: {summary}\n"
        f"TAGS: {', '.join(auto_tags)}\n"
    )
    embedding = embed_text(client, doc_text)

    reel_id = str(uuid.uuid4())

    # 5. Store in Pinecone
    index.upsert(
        vectors=[
            {
                "id": reel_id,
                "values": embedding,
                "metadata": {
                    "url": payload.url,
                    "summary": summary,
                    "tags": auto_tags,
                    "doc_text": doc_text,
                    **enrichment_metadata,
                    "enrichment_json": enrichment_json,
                },
            }
        ]
    )

    return AddReelResponse(
        reel_id=reel_id,
        summary=summary,
        auto_tags=auto_tags,
        enrichment=EnrichmentData(**enrichment),
    )


@app.post("/query", response_model=QueryResponse)
def query_reels(payload: QueryRequest):
    """
    Example queries:
    - 'show restaurants we saved in Goa'
    - 'any reels about Bali?'
    - 'cheap street food ideas?'
    """
    # 1. Embed query
    query_embedding = embed_text(client, payload.query)

    # 2. Vector search in Pinecone
    res = index.query(
        vector=query_embedding,
        top_k=payload.top_k,
        include_metadata=True,
    )

    if not res.matches:
        return QueryResponse(results=[])

    results: List[ReelResult] = []
    for match in res.matches:
        meta = match.metadata or {}
        score = float(match.score) if match.score is not None else 0.0
        enrichment = EnrichmentData(
            place_name=meta.get("enrichment_place_name"),
            city=meta.get("enrichment_city"),
            country=meta.get("enrichment_country"),
            lat=meta.get("enrichment_lat"),
            lng=meta.get("enrichment_lng"),
            rating=meta.get("enrichment_rating"),
            category=meta.get("enrichment_category"),
            cuisine=meta.get("enrichment_cuisine"),
            price_range=meta.get("enrichment_price_range"),
            summary=meta.get("enrichment_summary", meta.get("summary", "")),
            tags=meta.get("enrichment_tags", []),
        )
        results.append(
            ReelResult(
                reel_id=match.id,
                url=meta.get("url", ""),
                summary=meta.get("summary", ""),
                auto_tags=meta.get("tags", []),
                score=score,
                enrichment=enrichment,
            )
        )

    return QueryResponse(results=results)


@app.post("/enrich", response_model=EnrichResponse)
async def enrich_summary(payload: EnrichRequest):
    enrichment = await enrich_reel_summary(payload.vision_summary)
    return EnrichResponse(enrichment=EnrichmentData(**enrichment))

