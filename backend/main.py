import os
import uuid
import logging
from typing import List

from fastapi import BackgroundTasks, FastAPI, HTTPException
from dotenv import load_dotenv

from openai import OpenAI
from pinecone import Pinecone

from .helpers import embed_text, enrich_reel_summary, strip_igsh_parameter
from .job_store import get_job_status, set_job_status
from .models import (
    AgenticQueryResponse,
    AddReelRequest,
    AddReelResponse,
    CreateIngestionResponse,
    EnrichRequest,
    EnrichResponse,
    EnrichmentData,
    IngestionStatus,
    IngestionStatusResponse,
    QueryRequest,
    QueryResponse,
    ReelResult,
)
from .query_orchestrator import handle_query_agentic
from .services.ingestion_worker import process_reel_ingestion
from .tasks import process_reel_ingestion_task

load_dotenv()
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

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
app = FastAPI(title="Travel Reels Knowledge Base")


@app.post("/reels", response_model=CreateIngestionResponse, status_code=202)
async def add_reel(payload: AddReelRequest, background_tasks: BackgroundTasks):
    reel_url = strip_igsh_parameter(payload.url)
    if not reel_url:
        raise HTTPException(status_code=400, detail="A valid reel URL is required.")

    job_id = str(uuid.uuid4())
    set_job_status(
        job_id,
        {
            "job_id": job_id,
            "status": IngestionStatus.queued.value,
            "stage": "queued",
            "reel_url": reel_url,
        },
    )
    try:
        process_reel_ingestion_task.delay(job_id, reel_url, payload.manual_tags)
        logger.info("ingestion_enqueued backend=celery job_id=%s", job_id)
    except Exception as ex:
        logger.warning("celery_enqueue_failed fallback=background_tasks reason=%s", ex)
        background_tasks.add_task(process_reel_ingestion, job_id, reel_url, payload.manual_tags)

    return CreateIngestionResponse(job_id=job_id, status=IngestionStatus.queued)


@app.get("/reels/{job_id}/status", response_model=IngestionStatusResponse)
async def get_reel_ingestion_status(job_id: str):
    data = get_job_status(job_id)
    if not data:
        raise HTTPException(status_code=404, detail=f"Unknown job_id: {job_id}")

    result_payload = data.get("result")
    parsed_result = None
    if isinstance(result_payload, dict):
        try:
            parsed_result = AddReelResponse(**result_payload)
        except Exception:
            parsed_result = None
    return IngestionStatusResponse(
        job_id=job_id,
        status=IngestionStatus(data["status"]),
        stage=data.get("stage"),
        reel_id=data.get("reel_id"),
        error=data.get("error"),
        result=parsed_result,
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
                url=strip_igsh_parameter(meta.get("url", "")),
                summary=meta.get("summary", ""),
                auto_tags=meta.get("tags", []),
                score=score,
                enrichment=enrichment,
            )
        )

    return QueryResponse(results=results)


@app.post("/query-agentic", response_model=AgenticQueryResponse)
async def query_reels_agentic(payload: QueryRequest):
    response = await handle_query_agentic(
        query=payload.query,
        top_k=payload.top_k,
        client=client,
        index=index,
        conversation_history=payload.conversation_history,
    )
    logger.info(
        "query_agentic intent=%s route=%s confidence=%.2f results=%d map_points=%d",
        response.intent.value,
        response.meta.debug_route,
        response.meta.confidence,
        response.meta.result_count,
        response.meta.map_points_count,
    )
    return response


@app.post("/enrich", response_model=EnrichResponse)
async def enrich_summary(payload: EnrichRequest):
    enrichment = await enrich_reel_summary(payload.vision_summary)
    return EnrichResponse(enrichment=EnrichmentData(**enrichment))

